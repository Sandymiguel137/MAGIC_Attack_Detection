import scipy.io as sio
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import scipy.linalg as spl

class data_loader_FDIPhy(object):
    def __init__(self, file_path1, file_path2):
        self.data_input_beforeSE = np.load(file_path1)  # shape: (num_time, num_bus, num_sample)
        self.label_truth = np.load(file_path2)
        #self.data_0_rec = self.data_input_beforeSE[0, :]
        #self.data_0_tru = self.label_truth[0, :]
        # If no SE and no FDI, then use true voltage phasors
        self.data_recover = self.data_input_beforeSE.copy()
        
    def next_state(self, time_counter_input, time_counter_output):
        #data_feature = self.data_input_beforeSE[time_counter_input, :]  # before state estimation
        data_feature = self.data_recover[time_counter_input, :]  # after state estimation
        label_truth = self.label_truth[time_counter_output, :]
        return label_truth, data_feature

    def state_estimation(self, Y_norm_sparse, sensor_location_indices):
        num_time, num_bus, num_sample = self.data_input_beforeSE.shape
        Y_full = Y_norm_sparse.toarray()
        mask_available_sensors = np.zeros(num_bus, dtype=bool)
        mask_available_sensors[sensor_location_indices] = True
        sensor_indices = np.where(mask_available_sensors)[0]
        num_sensor = len(sensor_indices)
        nonsensor_indices = np.where(~mask_available_sensors)[0]
        new_bus_order = np.concatenate([sensor_indices, nonsensor_indices])
        reorder_inv = np.argsort(new_bus_order)
        #print('num_sensor', num_sensor)
        #print("mask available sensor:", mask_available_sensors)
        #print("sensor bus (ascending order):", sensor_indices)
        #print("non sensor bus (ascending order):", nonsensor_indices)
        #print("new node order (first sensor, then non-sensor):", new_bus_order)
        #print("reorder index to recover original node order:", reorder_inv)
        Y_AA = Y_full[mask_available_sensors, :][:, mask_available_sensors]
        Y_AU = Y_full[mask_available_sensors, :][:, ~mask_available_sensors]
        Y_UA = Y_full[~mask_available_sensors, :][:, mask_available_sensors]
        Y_UU = Y_full[~mask_available_sensors, :][:, ~mask_available_sensors]
        Y_block = np.block([[Y_AA, Y_AU],
                            [Y_UA, Y_UU]])
        system_matrix = np.vstack((
            np.hstack((Y_AA, Y_AU)),
            np.hstack((np.eye(num_sensor, dtype=complex),
                       np.zeros((num_sensor, num_bus - num_sensor), dtype=complex)))
        ))
        mmse_regularization_factor = 0.004
        state_estimate_projector = (np.linalg.pinv(
            system_matrix.T.conjugate() @ system_matrix 
            + mmse_regularization_factor * Y_block,
            rcond=1e-8
        ) @ system_matrix.T.conjugate())
        #print('system_matrix shape:', system_matrix.shape)
        #print('state_estimate_projector shape:', state_estimate_projector.shape)
        self.data_recover = np.zeros((num_time, num_bus, num_sample), dtype=complex)       
        
        for t in range(num_time):
            for s in range(num_sample):
                voltage_vector = self.data_input_beforeSE[t, :, s]
                voltage_ordered = np.concatenate([voltage_vector[mask_available_sensors],
                                                  voltage_vector[~mask_available_sensors]])
                voltage_ordered = voltage_ordered.reshape(-1, 1)  
                measurement = system_matrix @ voltage_ordered
                state_est = state_estimate_projector @ measurement
                state_est_orig = state_est[reorder_inv, :]
                flatten_state_est_orig = state_est_orig.flatten()
                self.data_recover[t, :, s] = flatten_state_est_orig

        #diff = self.data_input_beforeSE - self.data_recover
        #print('diff shape:', diff.shape)
        #print(diff[0, :, 0])
        #print('size of GCN input after SE', self.data_recover.shape)

    def state_est_withFDI(self, Y_norm_sparse, sensor_location_indices):
        num_time, num_bus, num_sample = self.data_input_beforeSE.shape
        Y_full = Y_norm_sparse.toarray()
        mask_available_sensors = np.zeros(num_bus, dtype=bool)
        mask_available_sensors[sensor_location_indices] = True
        sensor_indices = np.where(mask_available_sensors)[0]
        num_sensor = len(sensor_indices)
        nonsensor_indices = np.where(~mask_available_sensors)[0]
        new_bus_order = np.concatenate([sensor_indices, nonsensor_indices])
        reorder_inv = np.argsort(new_bus_order)

        # reorder the Y matrix
        Y_AA = Y_full[mask_available_sensors, :][:, mask_available_sensors]
        Y_AU = Y_full[mask_available_sensors, :][:, ~mask_available_sensors]
        Y_UA = Y_full[~mask_available_sensors, :][:, mask_available_sensors]
        Y_UU = Y_full[~mask_available_sensors, :][:, ~mask_available_sensors]
        Y_block = np.block([[Y_AA, Y_AU],
                            [Y_UA, Y_UU]])
        system_matrix = np.vstack((
            np.hstack((Y_AA, Y_AU)),
            np.hstack((np.eye(num_sensor, dtype=complex),
                    np.zeros((num_sensor, num_bus - num_sensor), dtype=complex)))
        ))
        mmse_regularization_factor = 0.004
        state_estimate_projector = (np.linalg.pinv(
            system_matrix.T.conjugate() @ system_matrix 
            + mmse_regularization_factor * Y_block,
            rcond=1e-8
        ) @ system_matrix.T.conjugate())
        
        # init data_recover
        self.data_recover = np.zeros((num_time, num_bus, num_sample), dtype=complex)
        # attack_labels: (binary 1: with FDI, 0: no FDI)
        self.attack_labels = np.zeros((num_time, num_sensor), dtype=int)

        p = 1 / num_sensor
        
        for t in range(num_time):
            # generate FDI attack
            # geometric distribution to randomly generate attackers
            #random_integer0 = np.random.geometric(p) - 1
            random_integer = min(np.random.geometric(p) - 1, num_sensor)
            # uniform distribution to randomly generate attackers
            #random_integer = np.random.randint(1, num_sensor)

            if random_integer > 0:
                ell_ind = np.random.choice(num_sensor, random_integer, replace=False)
            else:
                ell_ind = np.array([], dtype=int)
            mask_ell = np.zeros(num_sensor, dtype=bool)
            mask_ell[ell_ind] = True
            mask_honest = ~mask_ell

            # if FDI exists
            if np.sum(mask_honest) > 0 and np.sum(mask_ell) > 0:
                Y_attack = Y_AA[mask_honest, :][:, mask_ell]
            else:
                Y_attack = np.empty((0, 0))
            
            if Y_attack.size != 0 and Y_attack.shape[0] > 0 and Y_attack.shape[1] > 0:
                Null_space = spl.null_space(Y_attack)
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
                # sensor with/without FDI
                voltage_sensor_attacked = voltage_vector[mask_available_sensors] + attack_vec_da.flatten()
                voltage_nonsensor = voltage_vector[~mask_available_sensors]
                voltage_ordered = np.concatenate([voltage_sensor_attacked, voltage_nonsensor]).reshape(-1, 1)
                measurement = system_matrix @ voltage_ordered
                state_est = state_estimate_projector @ measurement
                state_est_orig = state_est[reorder_inv, :]
                self.data_recover[t, :, s] = state_est_orig.flatten()
        
        #print('size of label_truth before FDI:', self.label_truth.shape)
        #print('size of attack_labels:', self.attack_labels.shape)
        # label_truth: label of inverter physical attack
        # attack_labels: label of FDI attack
        self.label_truth = np.concatenate((self.label_truth, self.attack_labels), axis=1)
        #print('size of label_truth after FDI:', self.label_truth.shape)

class data_loader_phy_st(object):
    def __init__(self, file_path1, file_path2):
        self.data_input_beforeSE = np.load(file_path1)  # shape: (num_time, num_bus, num_sample)
        self.label_truth = np.load(file_path2)
        self.data_0_rec = self.data_input_beforeSE[0, :]
        self.data_0_tru = self.label_truth[0, :]
        self.data_recover = None
        
    def next_state(self, time_counter_input, time_counter_output):
        #data_feature = self.data_input_beforeSE[time_counter_input, :]  # before state estimation
        data_feature = self.data_recover[time_counter_input, :]  # after state estimation
        label_truth = self.label_truth[time_counter_output, :]
        return label_truth, data_feature

    def state_estimation(self, Y_norm_sparse, sensor_location_indices, noise_std=0.001, mmse_regularization_factor=0.004):
        num_time, num_bus, num_sample = self.data_input_beforeSE.shape
        Y_full = Y_norm_sparse.toarray()
        mask_available_sensors = np.zeros(num_bus, dtype=bool)
        mask_available_sensors[sensor_location_indices] = True
        sensor_indices = np.where(mask_available_sensors)[0]
        num_sensor = len(sensor_indices)
        nonsensor_indices = np.where(~mask_available_sensors)[0]
        new_bus_order = np.concatenate([sensor_indices, nonsensor_indices])
        reorder_inv = np.argsort(new_bus_order)
        #print('num_sensor', num_sensor)
        #print("mask available sensor:", mask_available_sensors)
        #print("sensor bus (ascending order):", sensor_indices)
        #print("non sensor bus (ascending order):", nonsensor_indices)
        #print("new node order (first sensor, then non-sensor):", new_bus_order)
        #print("reorder index to recover original node order:", reorder_inv)
        Y_AA = Y_full[mask_available_sensors, :][:, mask_available_sensors]
        Y_AU = Y_full[mask_available_sensors, :][:, ~mask_available_sensors]
        Y_UA = Y_full[~mask_available_sensors, :][:, mask_available_sensors]
        Y_UU = Y_full[~mask_available_sensors, :][:, ~mask_available_sensors]
        Y_block = np.block([[Y_AA, Y_AU],
                            [Y_UA, Y_UU]])
        system_matrix = np.vstack((
            np.hstack((Y_AA, Y_AU)),
            np.hstack((np.eye(num_sensor, dtype=complex),
                       np.zeros((num_sensor, num_bus - num_sensor), dtype=complex)))
        ))
        state_estimate_projector = (np.linalg.pinv(
            system_matrix.T.conjugate() @ system_matrix 
            + mmse_regularization_factor * Y_block,
            rcond=1e-8
        ) @ system_matrix.T.conjugate())
        #print('system_matrix shape:', system_matrix.shape)
        #print('state_estimate_projector shape:', state_estimate_projector.shape)
        self.data_recover = np.zeros((num_time, num_bus, num_sample), dtype=complex)
        
        for t in range(num_time):
            for s in range(num_sample):
                voltage_vector = self.data_input_beforeSE[t, :, s]
                voltage_ordered = np.concatenate([voltage_vector[mask_available_sensors],
                                                  voltage_vector[~mask_available_sensors]])
                voltage_ordered = voltage_ordered.reshape(-1, 1)  
                measurement = system_matrix @ voltage_ordered
                state_est = state_estimate_projector @ measurement
                state_est_orig = state_est[reorder_inv, :]
                flatten_state_est_orig = state_est_orig.flatten()
                self.data_recover[t, :, s] = flatten_state_est_orig

        #diff = self.data_input_beforeSE - self.data_recover
        #print('diff shape:', diff.shape)
        #print(diff[0, :, 0])
        #print('size of GCN input after SE', self.data_recover.shape)


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
