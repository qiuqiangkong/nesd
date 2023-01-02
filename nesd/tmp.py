class Model_06a_unet_sep(nn.Module, Base):
    def __init__(self, microphones_num, classes_num):
        super(Model_06a_unet_sep, self).__init__() 

        window_size = 512
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 6

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.positional_encoder = PositionalEncoder(factor=positional_embedding_factor)
        self.positional_encoder_room_xyz = PositionalEncoderRoomXYZ(factor=positional_embedding_factor)

        self.pre_value_cnn = nn.Conv2d(in_channels=microphones_num + 1, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)
        self.pre_angle_cnn = nn.Conv2d(in_channels=12, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)

        self.input_value_cnn1 = EncoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn2 = EncoderBlockRes1B(in_channels=32, out_channels=64, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn3 = EncoderBlockRes1B(in_channels=64, out_channels=128, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        # self.input_value_cnn4 = EncoderBlockRes1B(in_channels=128, out_channels=256, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_fc1 = nn.Linear(in_features=32 * 128, out_features=1024)

        self.input_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4) * microphones_num, out_features=128, bias=True)
        self.input_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6) * microphones_num, out_features=128, bias=True)

        self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        self.fc_final_tracking = nn.Linear(1024, 1, bias=True)

        # Sep
        self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

        self.target_sources_num = 1
        self.output_channels = 1
        self.K = 3

    def init_weights(self):

        init_layer(self.pre_value_cnn)
        init_layer(self.pre_angle_cnn)
        init_layer(self.input_value_fc1)

        init_layer(self.input_angle_fc)
        init_layer(self.input_pos_xyz_fc)
        init_layer(self.output_angle_fc)
        init_layer(self.output_pos_xyz_fc)

        init_layer(self.fc_inter)
        init_layer(self.fc_final_tracking)

        init_layer(self.dec_final)

    def pad(self, x, window_size, hop_size):
        # To make show the center of fft window is the center of a frame
        x = F.pad(x, pad=(window_size // 2 - hop_size // 2, window_size // 2 - hop_size // 2), mode=self.pad_mode)
        return x

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, data_dict, max_sep_rays=None, do_sep=True, **kwargs):

        eps = 1e-10

        omni_waveform = torch.sum(data_dict['mic_waveform'], dim=1, keepdim=True)
        omni_mag0, omni_cos0, omni_sin0 = self.wav_to_spectrogram_phase(omni_waveform, eps) # (batch_size, 1, L, F)

        mic_waveform = data_dict['mic_waveform']
        mic_direction = data_dict['mic_direction']
        mic_position = data_dict['mic_position']


        _, mic_di_azimuth, mic_di_zenith = cart2sph_torch(
            x=mic_direction[:, :, 0], 
            y=mic_direction[:, :, 1], 
            z=mic_direction[:, :, 2]
        )

        ray_direction = data_dict['ray_direction']
        ray_position = data_dict['ray_origin']

        _, ray_di_azimuth, ray_di_zenith = cart2sph_torch(
            x=ray_direction[:, :, 0], 
            y=ray_direction[:, :, 1], 
            z=ray_direction[:, :, 2]
        )

        # positional encoding
        mic_pos_emb = self.positional_encoder(
            azimuth=mic_di_azimuth,
            elevation=mic_di_zenith,
        )
        # (batch_size, mics_num, outputs_num)

        mic_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=mic_position[:, :, 0],
            y=mic_position[:, :, 1],
            z=mic_position[:, :, 2],
        )
        
        ray_pos_emb = self.positional_encoder(
            azimuth=ray_di_azimuth,
            elevation=ray_di_zenith,
        )
        # (batch_size, mics_num, outputs_num)

        ray_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=ray_position[:, :, 0],
            y=ray_position[:, :, 1],
            z=ray_position[:, :, 2],
        )

        batch_size = omni_waveform.shape[0]

        omni_mag, omni_cos, omni_sin = self.wav_to_spectrogram_phase(omni_waveform, eps) # (batch_size, 1, L, F)
        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, eps) # (batch_size, 1, L, F)

        omni_real = omni_mag * omni_cos
        omni_imag = omni_mag * omni_sin

        delta_cos_list = []
        delta_sin_list = []

        for i in range(1, 4):
            for j in range(0, i):
                _delta_cos = mic_cos[:, i, :, :] * mic_cos[:, j, :, :] + \
                    mic_sin[:, i, :, :] * mic_sin[:, j, :, :]

                _delta_sin = mic_sin[:, i, :, :] * mic_cos[:, j, :, :] - \
                    mic_cos[:, i, :, :] * mic_sin[:, j, :, :]

                delta_cos_list.append(_delta_cos)
                delta_sin_list.append(_delta_sin)

        delta_cos = torch.stack(delta_cos_list, dim=1)
        delta_sin = torch.stack(delta_sin_list, dim=1)

        delta_real = omni_mag * delta_cos
        delta_imag = omni_mag * delta_sin

        x1 = torch.cat((omni_mag, mic_mag), dim=1)
        x2 = torch.cat((delta_cos, delta_sin), dim=1)

        x1 = self.pre_value_cnn(x1)
        x2 = self.pre_angle_cnn(x2)
        
        x = torch.cat((x1, x2), dim=1)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)
        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        # input value FC
        enc1_pool, enc1 = self.input_value_cnn1(x)
        enc2_pool, enc2 = self.input_value_cnn2(enc1_pool)
        enc3_pool, enc3 = self.input_value_cnn3(enc2_pool)
        # a1 = self.input_value_cnn4(a1)
        a1 = enc3_pool

        a1 = a1.permute(0, 2, 1, 3)
        a1 = a1.flatten(2)

        a1 = F.leaky_relu_(self.input_value_fc1(a1), negative_slope=0.01)

        # input angle FC
        a2 = mic_pos_emb.flatten(1)
        a2 = F.leaky_relu_(self.input_angle_fc(a2))    # (batch_size, 1, T', C)        

        a2_xyz = mic_pos_xyz_emb.flatten(1)
        a2_xyz = F.leaky_relu_(self.input_pos_xyz_fc(a2_xyz))

        frames_num = 40
        a2 = torch.tile(a2[:, None, :], (1, frames_num, 1))
        a2_xyz = torch.tile(a2_xyz[:, None, :], (1, frames_num, 1))
        x = torch.cat((a1, a2, a2_xyz), dim=-1) # (batch_size, T, C * 2)

        rays_num = ray_direction.shape[1]

        x = torch.tile(x[:, None, :, :], (1, rays_num, 1, 1))
        # (batch_size, outputs_num, T, C * 2)

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_pos_emb))
        a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)

        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)

            output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)

        if do_sep:
            if max_sep_rays:
                x = inter_emb[:, 0 : max_sep_rays, :, :]
            else:
                x = inter_emb

            x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_sep_rays, :, :]))

            batch_size, sep_rays_num, _T, _C = x.shape
            x = x.reshape(batch_size * sep_rays_num, _T, 128, 32)
            x = x.permute(0, 2, 1, 3)

            if max_sep_rays:
                enc1 = self.tile_omni(x=enc1, rays_num=max_sep_rays)
                enc2 = self.tile_omni(x=enc2, rays_num=max_sep_rays)
                enc3 = self.tile_omni(x=enc3, rays_num=max_sep_rays)
            else:
                enc1 = self.tile_omni(x=enc1, rays_num=inter_emb.shape[1])
                enc2 = self.tile_omni(x=enc2, rays_num=inter_emb.shape[1])
                enc3 = self.tile_omni(x=enc3, rays_num=inter_emb.shape[1])

            x = self.dec_block1(x, enc3)
            x = self.dec_block2(x, enc2)
            x = self.dec_block3(x, enc1)
            x = self.dec_final(x)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

            x = x[:, :, 0:origin_len, :]

            audio_length = omni_waveform.shape[2]

            omni_mag0 = self.tile_omni(omni_mag0, sep_rays_num)
            omni_cos0 = self.tile_omni(omni_cos0, sep_rays_num)
            omni_sin0 = self.tile_omni(omni_sin0, sep_rays_num)

            separated_audio = self.feature_maps_to_wav(
                input_tensor=x,
                sp=omni_mag0,
                sin_in=omni_sin0,
                cos_in=omni_cos0,
                audio_length=audio_length,
            )

            separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)

            output_dict['ray_waveform'] = separated_audio

        return output_dict

    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)