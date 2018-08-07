import os
import pickle

from baselines import logger
from mpi4py import MPI


class Recorder(object):
    def __init__(self, nenvs, nlumps):
        self.nenvs = nenvs
        self.nlumps = nlumps
        self.nenvs_per_lump = nenvs // nlumps
        self.acs = [[] for _ in range(nenvs)]
        self.int_rews = [[] for _ in range(nenvs)]
        self.ext_rews = [[] for _ in range(nenvs)]
        self.ep_infos = [{} for _ in range(nenvs)]
        self.filenames = [self.get_filename(i) for i in range(nenvs)]
        all_filenames = MPI.COMM_WORLD.allreduce(self.filenames, MPI.SUM)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("videos saved to ", self.filenames[0])
            print(all_filenames)
            print(len(all_filenames))
            with open(os.path.join(logger.get_dir(), "all_filenames.pkl"), "wb") as f:
                pickle.dump(all_filenames, f)

    def record(self, timestep, lump, acs, infos, int_rew, ext_rew, news):
        for out_index in range(self.nenvs_per_lump):
            in_index = out_index + lump * self.nenvs_per_lump
            if timestep == 0:
                self.acs[in_index].append(acs[out_index])
            else:
                if self.is_first_episode_step(in_index):
                    try:
                        self.ep_infos[in_index]['random_state'] = infos[out_index]['random_state']
                    except:
                        pass

                self.int_rews[in_index].append(int_rew[out_index])
                self.ext_rews[in_index].append(ext_rew[out_index])

                if news[out_index]:
                    self.ep_infos[in_index]['ret'] = infos[out_index]['episode']['r']
                    self.ep_infos[in_index]['len'] = infos[out_index]['episode']['l']
                    if 'mz_episode' in infos[out_index]:
                        self.ep_infos[in_index]['rooms'] = infos[out_index]['mz_episode']['visited_rooms']
                        self.mz = True
                    else:
                        self.mz = False
                    # assert self.ep_infos[in_index]['ret'] == sum(self.ext_rews[in_index])
                    self.dump_episode(in_index)

                self.acs[in_index].append(acs[out_index])

    def dump_episode(self, i):
        episode = {'acs': self.acs[i],
                   'int_rew': self.int_rews[i],
                   'info': self.ep_infos[i]}
        filename = self.filenames[i]
        if self.episode_worth_saving(i):
            with open(filename, 'ab') as f:
                pickle.dump(episode, f, protocol=-1)
        self.acs[i].clear()
        self.int_rews[i].clear()
        self.ext_rews[i].clear()
        self.ep_infos[i].clear()

    def episode_worth_saving(self, i):
        if self.mz:
            return (i == 0 and MPI.COMM_WORLD.Get_rank() == 0) or (
                any(room in self.ep_infos[i]["rooms"] for room in list(range(3, 5)) + list(range(8, 25))))
        else:
            return (i == 0 and MPI.COMM_WORLD.Get_rank() == 0)

    def is_first_episode_step(self, i):
        return len(self.int_rews[i]) == 0

    def get_filename(self, i):
        filename = os.path.join(logger.get_dir(), 'env{}_{}.pk'.format(MPI.COMM_WORLD.Get_rank(), i))
        return filename
