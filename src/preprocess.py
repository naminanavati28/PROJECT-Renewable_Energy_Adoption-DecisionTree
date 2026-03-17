import pandas as pd

def prepare_input(carbon_emissions, energy_output, renewability_index, cost_efficiency):
    """
    Prepares a single-row DataFrame containing the input values
    in the exact same order and with the same feature names
    used during model training.
    """
    data = {
        'carbon_emissions': [carbon_emissions],
        'energy_output': [energy_output],
        'renewability_index': [renewability_index],
        'cost_efficiency': [cost_efficiency]
    }

    return pd.DataFrame(data)
