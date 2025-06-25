import scipy.io as sio
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import scipy.linalg as spl
from measurements import measurements
import numpy as np
import scipy.linalg as spl
from measurements import measurements, get_per_unit_current_map_from_Ybus

class data_loader_FDIPhy(object):
    def __init__(self, file_path1, file_path2, Y_NodeOrder=None):
        self.data_input_beforeSE = np.load(file_path1)
        self.label_truth = np.load(file_path2)
        self.data_recover = self.data_input_beforeSE.copy()
        self.Y_NodeOrder = Y_NodeOrder

    def next_state(self, time_counter_input, time_counter_output):
        data_feature = self.data_recover[time_counter_input, :]
        label_truth = self.label_truth[time_counter_output, :]
        return label_truth, data_feature

    def state_est_withFDI(self, Y_norm_sparse, sensor_location_indices, voltage_scale_factor=0.01):
        num_time, num_bus, num_sample = self.data_input_beforeSE.shape
        Y_full = Y_norm_sparse.toarray()

        mask_available_sensors = np.zeros(num_bus, dtype=bool)
        mask_available_sensors[sensor_location_indices] = True
        sensor_indices = np.where(mask_available_sensors)[0]
        num_sensor = len(sensor_indices)
        nonsensor_indices = np.where(~mask_available_sensors)[0]
        new_bus_order = np.concatenate([sensor_indices, nonsensor_indices])
        reorder_inv = np.argsort(new_bus_order)

        Y_AA = Y_full[mask_available_sensors, :][:, mask_available_sensors]
        Y_AU = Y_full[mask_available_sensors, :][:, ~mask_available_sensors]
        Y_UA = Y_full[~mask_available_sensors, :][:, mask_available_sensors]
        Y_UU = Y_full[~mask_available_sensors, :][:, ~mask_available_sensors]
        Y_block = np.block([[Y_AA, Y_AU], [Y_UA, Y_UU]])

        system_matrix = np.vstack((
            np.hstack((Y_AA, Y_AU)),
            np.hstack((np.eye(num_sensor, dtype=complex),
                       np.zeros((num_sensor, num_bus - num_sensor), dtype=complex)))
        ))
        mmse_regularization_factor = 0.004
        state_estimate_projector = np.linalg.pinv(
            system_matrix.T.conjugate() @ system_matrix + mmse_regularization_factor * Y_block,
            rcond=1e-8
        ) @ system_matrix.T.conjugate()

        self.data_recover = np.zeros((num_time, num_bus, num_sample), dtype=complex)
        self.attack_labels = np.zeros((num_time, num_sensor), dtype=int)

        p = 1 / num_sensor

        for t in range(num_time):
            random_integer = min(np.random.geometric(p) - 1, num_sensor)
            ell_ind = np.random.choice(num_sensor, random_integer, replace=False) if random_integer > 0 else np.array([], dtype=int)
            mask_ell = np.zeros(num_sensor, dtype=bool)
            mask_ell[ell_ind] = True
            mask_honest = ~mask_ell

            if np.sum(mask_honest) > 0 and np.sum(mask_ell) > 0:
                Y_attack = Y_AA[mask_honest, :][:, mask_ell]
                Null_space = spl.null_space(Y_attack) if Y_attack.size > 0 else np.empty((0, 0))
                if Null_space.shape[1] > 0:
                    attack_vec_dm = Null_space @ np.random.uniform(low=-0.005, high=0.005, size=(Null_space.shape[1], 1))
                    attack_vec_dm[np.abs(attack_vec_dm) < 1e-4] = 0
                else:
                    attack_vec_dm = np.zeros((0, 1), dtype=complex)
            else:
                attack_vec_dm = np.zeros((0, 1), dtype=complex)

            attack_vec_da = np.zeros((num_sensor, 1), dtype=complex)
            if attack_vec_dm.size > 0:
                attack_vec_da[mask_ell] = attack_vec_dm

            index_label_attack = (attack_vec_da.flatten() != 0)
            self.attack_labels[t, index_label_attack] = 1

            for s in range(num_sample):
                voltage_vector = self.data_input_beforeSE[t, :, s]
                voltage_nonsensor = voltage_vector[~mask_available_sensors]
                sensor_nodes = [self.Y_NodeOrder[i] for i in sensor_location_indices]
                # Clean input from OpenDSS PF result
                volt_dict_clean = {
                    sensor_nodes[i].lower(): voltage_vector[sensor_location_indices[i]]
                    for i in range(num_sensor)
                }

                # Inject multiplicative noise via measurements()
                current_dict = get_per_unit_current_map_from_Ybus()
                z_dict = measurements(sensor_nodes, [], volt_dict_clean, current_dict=current_dict, voltage_scale_factor=voltage_scale_factor)

                v_sensor_noisy = np.array([
                    z_dict[node.lower()][0] for node in sensor_nodes
                ], dtype=complex)

                # Inject stealthy FDI attack AFTER noise
                voltage_sensor_attacked = v_sensor_noisy + attack_vec_da.flatten()
                voltage_ordered = np.concatenate([voltage_sensor_attacked, voltage_nonsensor]).reshape(-1, 1)

                # MMSE estimation
                measurement = system_matrix @ voltage_ordered
                state_est = state_estimate_projector @ measurement
                state_est_orig = state_est[reorder_inv, :]
                self.data_recover[t, :, s] = state_est_orig.flatten()

        self.label_truth = np.concatenate((self.label_truth, self.attack_labels), axis=1)
        expected_dim = self.label_truth.shape[1]
        actual_dim = 30 + len(sensor_location_indices)
        assert expected_dim == actual_dim, f"Label dimension mismatch: got {expected_dim}, expected {actual_dim}"


class data_loader_phy(object):
    def __init__(self, file_path1, file_path2):
        self.data_recover = np.load(file_path1)
        self.label_truth = np.load(file_path2)
        self.data_0_rec = self.data_recover[0, :]
        self.data_0_tru = self.label_truth[0, :]
        #print(self.data_recover.shape)
        #print(self.label_truth.shape)
        
    def next_state(self, time_counter_input, time_counter_output):
        data_feature = self.data_recover[time_counter_input, :]
        label_truth = self.label_truth[time_counter_output, :]
        return label_truth, data_feature
    
    
class data_loader(object):
    def __init__(self, file_name):
        self.data_recover =   sio.loadmat(file_name)['Vphasor_mat']
        self.label_truth   =   sio.loadmat(file_name)['attack_row_mat']
        self.data_0_rec = self.data_recover[0, :]
        self.data_0_tru = self.label_truth[0, :]
    
    def next_state(self, time_counter_input, time_counter_output):
        data_feature = self.data_recover[time_counter_input, :]
        label_truth = self.label_truth[time_counter_output, :]
        return label_truth, data_feature

class RolloutStorage(object):
    def __init__(self, num_steps,  obs_shape_input, obs_shape_out,  device):
        self.observations = torch.zeros(num_steps,  *obs_shape_input,  dtype = torch.cfloat).to(device)
        self.target = torch.zeros(num_steps,  *obs_shape_out,  dtype = torch.float).to(device)
        self.step = 0
        self.num_steps = num_steps
    
    def insert(self, current_obs, current_target):
        self.observations[self.step].copy_(current_obs)
        self.target[self.step].copy_(current_target)
        self.step = (self.step + 1) % self.num_steps

    
    def batch_generator(self, mini_batch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[indices]
            target_batch       = self.target[indices]
            yield observations_batch, target_batch
