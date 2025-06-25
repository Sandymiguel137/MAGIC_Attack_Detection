# This code simulates noisy sensor measurements for voltage and current in a power distribution network using OpenDSS:

# Uses OpenDSS simulation results (voltages from power flow)
# Computes injected current at each node via the Ybus matrix
# Adds multiplicative Gaussian noise to the voltage
# Returns measurements for:
    # mu-PMU: (voltage, current)
    # power meters: (power, |V|, |I|)

import numpy as np
import opendssdirect as dss
from scipy.sparse import csc_matrix

def get_all_node_voltageReIm_map_afterPF():
    dss_map = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        voltages = dss.Bus.PuVoltage()
        phases = dss.Bus.Nodes()
        for i, ph in enumerate(phases):
            key = f"{bus_name.lower()}.{ph}"
            dss_map[key] = (voltages[2 * i], voltages[2 * i + 1])
    return dss_map

def get_per_unit_current_map_from_Ybus():
    Y_sparse = csc_matrix(dss.YMatrix.getYsparse()) 
    Y_NodeOrder = dss.Circuit.YNodeOrder()
    voltage_map = get_all_node_voltageReIm_map_afterPF()
    N = len(Y_NodeOrder)

    V = np.zeros(N, dtype=complex)
    for i, node in enumerate(Y_NodeOrder):
        key = node.lower()
        if key in voltage_map:
            re_raw, im_raw = voltage_map[key]
            re = float(np.asarray(re_raw).item())
            im = float(np.asarray(im_raw).item())
            V[i] = complex(re, im)
        else:
            raise KeyError(f"Voltage for node '{key}' not found in voltage_map.")

    I = Y_sparse @ V

    current_map = {}
    for i, node in enumerate(Y_NodeOrder):
        current_map[node.lower()] = (I[i].real, I[i].imag)

    return current_map

def apply_voltage_noise(v_complex, scale, return_factor=False):
    rand_factor = np.clip(np.random.normal(loc=0, scale=1), -1, 1) 
    factor = 1 + scale * rand_factor
    noisy_v = v_complex * factor
    return (factor if return_factor else noisy_v)

def measurements(S_mu_PMU, S_power_meter, volt_dict, current_dict=None, voltage_scale_factor=0.01):
    set_mu = set(x.lower() for x in S_mu_PMU)
    set_pm = set(x.lower() for x in S_power_meter)
    if set_mu & set_pm:
        raise ValueError(f"Sensor sets must be disjoint. Overlap: {set_mu & set_pm}")

    voltage_map = get_all_node_voltageReIm_map_afterPF()
    current_map = get_per_unit_current_map_from_Ybus()

    z_dict = {}
    matched_mu, matched_pm = [], []

    # Process mu-PMU measurements
    for sensor in S_mu_PMU:
        key = sensor.lower()
        if key in volt_dict:
            v_clean = volt_dict[key]
            i_clean = current_dict[key] if current_dict and key in current_dict else 0
            v_noisy = apply_voltage_noise(v_clean, voltage_scale_factor)
            z_dict[key] = (v_noisy, i_clean)
            matched_mu.append(key)

    # Process power meters (complex power)
    for sensor in S_power_meter:
        key = sensor.lower()
        if key in voltage_map and key in current_map:
            v = complex(*voltage_map[key])
            i = complex(*current_map[key])
            scale_factor = apply_voltage_noise(v, voltage_scale_factor, return_factor=True)
            v_noisy = v * scale_factor
            s_measured = v_noisy * np.conj(i)
            z_dict[key] = (s_measured, abs(v_noisy), abs(i))
            matched_pm.append(key)

    print(f"mu-PMU matched: {len(matched_mu)}/{len(S_mu_PMU)}")
    if len(matched_mu) < len(S_mu_PMU):
        print("Unmatched mu-PMU examples:", list(set_mu - set(matched_mu))[:5])

    print(f"PowerMeter matched: {len(matched_pm)}/{len(S_power_meter)}")
    if len(matched_pm) < len(S_power_meter):
        print("Unmatched PowerMeter examples:", list(set_pm - set(matched_pm))[:5])

    return z_dict
