class DatasetDcase2021Task3_MovC:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.speed_of_sound = 343.
        self.sample_rate = 24000
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.segment_seconds = 3.
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

        self.dcase_fps = 10
        self.nesd_fps = 100
        self.classes_num = classes_num

        self.segment_frames_10fps = int(self.segment_seconds * self.dcase_fps) + 1

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        self.hdf5s_dir = hdf5s_dir
        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
                'mixture': (channels, segments_num),
            }
        """

        # Init mic pos, source pos, target pos

        # Calculate mic signal 
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

        mic_center_position = np.array([4, 4, 2])
        mic_center_position = expand_along_time(mic_center_position, self.frames_num)
        
        mics = []

        for mic_meta in self.mics_meta:

            relative_mic_posision = np.array(sph2cart(
                r=mic_meta['radius'], 
                azimuth=mic_meta['azimuth'], 
                colatitude=mic_meta['colatitude']
            ))

            mic_position = mic_center_position + relative_mic_posision[None, :]

            mic_look_direction = normalize(relative_mic_posision)
            mic_look_direction = expand_along_time(mic_look_direction, self.frames_num)

            mic = Microphone(
                position=mic_position,
                look_direction=mic_look_direction,
                directivity=mic_meta['directivity'],
            )
            mics.append(mic)

        # --------- Simulate mic

        hdf5s_num = len(self.hdf5_names)
        hdf5_index = random_state.randint(hdf5s_num)

        hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[hdf5_index])

        with h5py.File(hdf5_path, 'r') as hf:

            audio_seconds = hf['waveform'].shape[-1] // self.sample_rate
            begin_second = random_state.uniform(low=0, high=audio_seconds - self.segment_seconds)
            begin_second = np.round(begin_second, decimals=1)
            end_second = begin_second + self.segment_seconds

            begin_sample = int(begin_second * self.sample_rate)
            end_sample = int(begin_sample + self.segment_samples)

            for mic_index, mic in enumerate(mics):
                segment = int16_to_float32(hf['waveform'][mic_index, begin_sample : end_sample])
                mic.waveform = segment

            begin_frame_10fps = int(begin_second * self.dcase_fps)
            end_frame_10fps = int(end_second * self.dcase_fps)
            

            frame_indexes_10fps = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            azimuths, colatitudes = dcase_phase_to_nesd_phase(
                azimuth=hf['azimuth'][:],
                elevation=hf['elevation'][:],
            )

        # Collect sources
        event_ids = set(event_indexes)

        events_dict = {}

        for event_id in event_ids:

            event = None

            for n in range(len(frame_indexes_10fps)):

                frame_index_10fps = frame_indexes_10fps[n]

                if event_indexes[n] == event_id and begin_frame_10fps <= frame_index_10fps <= end_frame_10fps:

                    if event is None:
                        event = {
                            'class_id': np.ones(self.segment_frames_10fps, dtype=np.int32) * np.nan,
                            'azimuth': np.ones(self.segment_frames_10fps) * np.nan,
                            'colatitude': np.ones(self.segment_frames_10fps) * np.nan,
                        }

                    relative_frame_10fps = frame_index_10fps - begin_frame_10fps

                    event['class_id'][relative_frame_10fps] = class_indexes[n]
                    event['azimuth'][relative_frame_10fps] = azimuths[n]
                    event['colatitude'][relative_frame_10fps] = colatitudes[n]

            if event:
                events_dict[event_id] = event

        # sources
        sources = []

        for event_id in events_dict.keys():

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=events_dict[event_id]['azimuth'],
                colatitude=events_dict[event_id]['colatitude'],
            ), axis=-1)
            
            agent_to_src_direction_dcase_fps = extend_dcase_frames_to_nesd_frames(
                x=agent_to_src_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            source_position = agent_to_src_direction_dcase_fps + agent_position

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=np.nan * np.zeros(self.segment_samples),
            )
            sources.append(source)

        # --------- Hard example new position agents

        half_angle = math.atan2(0.1, 1)

        agents = []

        for event_id in events_dict.keys():

            class_id_array = events_dict[event_id]['class_id']
            azimuth_array = events_dict[event_id]['azimuth'].copy()
            colatitude_array = events_dict[event_id]['colatitude'].copy()

            agent_see_source = np.zeros(self.segment_frames_10fps)
            agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

            for i in range(self.segment_frames_10fps):
                if not math.isnan(class_id_array[i]):
                    agent_see_source[i] = 1
                    agent_see_source_classwise[i, int(class_id_array[i])] = 1

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=azimuth_array,
                colatitude=colatitude_array
            ), axis=-1)

            agent_look_direction = sample_agent_look_direction2(
                agent_to_src_direction=agent_to_src_direction,
                half_angle=half_angle,
                random_state=random_state,
            )

            agent_look_direction = extend_look_direction(agent_to_src_direction)
            
            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent_see_source = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source_classwise, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.ones(self.segment_samples) * np.nan,
                see_source=agent_see_source,
                see_source_classwise=agent_see_source_classwise,
            )

            agents.append(agent)

        #
        for event_id in events_dict.keys():

            for _ in range(5):

                class_id_array = events_dict[event_id]['class_id']
                azimuth_array = events_dict[event_id]['azimuth'].copy()
                colatitude_array = events_dict[event_id]['colatitude'].copy()

                agent_see_source = np.zeros(self.segment_frames_10fps)
                agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

                # for i in range(self.segment_frames_10fps):
                #     if not math.isnan(class_id_array[i]):
                #         agent_see_source[i] = 1
                #         agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                _frame_candidates = np.where(~np.isnan(agent_to_src_direction[:, 0]))[0]
                _frame_index = random_state.choice(_frame_candidates)
                agent_look_direction = agent_to_src_direction[_frame_index]

                agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

                agent_look_direction = sample_agent_look_direction2(
                    agent_to_src_direction=agent_look_direction,
                    half_angle=half_angle,
                    random_state=random_state,
                )
                from IPython import embed; embed(using=False); os._exit(0)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            agent_see_source[i] = 1
                            agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_see_source = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )

                agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source_classwise, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )
                
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=agent_see_source,
                    see_source_classwise=agent_see_source_classwise,
                )

                agents.append(agent)

        #
        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

            satisfied = True

            for event_id in events_dict.keys():

                azimuth_array = events_dict[event_id]['azimuth']
                colatitude_array = events_dict[event_id]['colatitude']

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            satisfied = False

            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=np.zeros(self.frames_num),
                    see_source_classwise=np.zeros((self.frames_num, self.classes_num))
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
            'agent_see_source_classwise': np.array([agent.see_source_classwise for agent in agents[0 : self.max_agents_contain_waveform]]),
        }
        
        # Plot
        if False:
            azis, cols = [], []
            for ray in rays:
                _, azi, col = cart2sph(
                    x=ray.direction[0],
                    y=ray.direction[1],
                    z=ray.direction[2],
                )
                azis.append(azi)
                cols.append(col)
            plt.scatter(azis, cols, s=4, c='b')

            azis, cols = [], []
            for source in sources:
                newpos_to_src = source.position - new_position
                _, azi, col = cart2sph(
                    x=newpos_to_src[0],
                    y=newpos_to_src[1],
                    z=newpos_to_src[2],
                )
                azis.append(azi)
                cols.append(col)
            plt.scatter(azis, cols, s=20, c='r', marker='+')
                
            plt.xlim(0, 2 * math.pi)
            plt.ylim(0, math.pi)
            plt.savefig('_zz.pdf')
            from IPython import embed; embed(using=False); os._exit(0)

        return data_dict


def sample_agent_look_direction2(agent_to_src_direction, half_angle, random_state):

    frames_num = agent_to_src_direction.shape[0]
    
    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    _, agent_to_src_azimuth, agent_to_src_colatitude = cart2sph(
        x=agent_to_src_direction[:, 0], 
        y=agent_to_src_direction[:, 1], 
        z=agent_to_src_direction[:, 2],
    )

    agent_look_azimuth = np.zeros(frames_num)
    agent_look_colatitude = np.zeros(frames_num)

    for i in range(frames_num):

        rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
            azimuth=agent_to_src_azimuth[i],
            colatitude=agent_to_src_colatitude[i],
        )

        agent_look_azimuth[i], agent_look_colatitude[i] = Rotator3D.rotate_azimuth_colatitude(
            rotation_matrix=rotation_matrix,
            azimuth=_azimuth,
            colatitude=_colatitude,
        )

    agent_look_direction = np.stack(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ), axis=-1)

    return agent_look_direction