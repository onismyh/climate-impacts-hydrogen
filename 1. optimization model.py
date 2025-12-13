from gurobipy import Model, GRB

# Function to perform optimization for each grid point
def optimize_for_grid(wind_output, pv_output, target_hydrogen_production, wind_cost, pv_cost, battery_cost, electrolyzer_cost, hydrogen_storage_cost,
                       wind_om, pv_om, battery_om, electrolyzer_om, hydrogen_storage_om, discount_rate, wind_lifetime, pv_lifetime, hydrogen_storage_lifetime,
                       battery_lifetime, electrolyzer_lifetime, surplus_penalty, battery_charge_penalty, battery_discharge_penalty, Flex_down,
                       electrolyzer_eff, battery_eff, hydrogen_storage_elec):
    """
    Optimizes the capacities of wind, PV, battery, electrolyzer, and hydrogen storage to minimize the levelized cost of hydrogen production while meeting a target annual hydrogen production.
    Parameters:
    - wind_output: Array of wind power output (per unit capacity) over time
    - pv_output: Array of PV power output (per unit capacity) over time
    - target_hydrogen_production: Target annual hydrogen production (reference demand: 1kg)
    - wind_cost, pv_cost, battery_cost, electrolyzer_cost, hydrogen_storage_cost: Capital costs ($/kW or $/kWh)
    - wind_om, pv_om, battery_om, electrolyzer_om, hydrogen_storage_om: Annual O&M costs ($/kW or $/kWh)
    - discount_rate: Discount rate for annualizing costs
    - wind_lifetime, pv_lifetime, battery_lifetime, electrolyzer_lifetime, hydrogen_storage_lifetime: Lifetimes (years)
    - surplus_penalty: Penalty cost for surplus energy ($/kWh)
    - battery_charge_penalty: Penalty cost for battery charging ($/kWh, 1e-6 recommended)
    - battery_discharge_penalty: Penalty cost for battery discharging ($/kWh, 1e-6 recommended)
    - Flex_down: Maximum allowable decrease in electrolyzer power per time step (as a fraction of capacity)
    - electrolyzer_eff: Electrolyzer efficiency (fraction)
    - battery_eff: Battery round-trip efficiency (fraction)
    - hydrogen_storage_elec: Electricity required per kg of hydrogen stored (kWh/kg, optional)
    Returns:
    - Dictionary with optimized capacities and levelized cost of hydrogen production
    """
    
    
    
    # Create Gurobi model
    model = Model("HydrogenProductionOptimization")
    model.setParam('OutputFlag', 0)
    
    # Set model parameters to handle large models
    model.setParam('TimeLimit', 100)  # Set a time limit of 100 seconds to avoid long runtimes
    model.setParam('MIPFocus', 1)  # Focus on finding a feasible solution
    model.setParam('Threads', 22)  # Limit the number of threads to reduce resource usage
    model.setParam('MIPGap', 0.1) 

    # Define decision variables
    wind_capacity = model.addVar(lb=0, name="wind_capacity")
    pv_capacity = model.addVar(lb=0, name="pv_capacity")
    battery_capacity = model.addVar(lb=0, name="battery_capacity")
    electrolyzer_capacity = model.addVar(lb=0, name="electrolyzer_capacity")
    hydrogen_storage_capacity = model.addVar(lb=0, name="hydrogen_storage_capacity")

    # Add energy balance variables for storage (aggregated to reduce model size)
    time_steps = 8760
    energy_balance = model.addVars(time_steps, lb=0, name="energy_balance")

    # Add variables for battery charging and discharging (aggregated to reduce model size)
    battery_charge = model.addVars(time_steps, lb=0, name="battery_charge")
    battery_discharge = model.addVars(time_steps, lb=0, name="battery_discharge")
    battery_hours = 4  # Number of hours for battery storage

    # Add variables for electrolyzer operation, surplus, and unmet demand
    electrolyzer_power = model.addVars(time_steps, lb=0, name="electrolyzer_power")
    surplus = model.addVars(time_steps, lb=0, name="surplus")

    hydrogen_storage_balance = model.addVars(time_steps, lb=0, name="hydrogen_storage_balance")
    hydrogen_charge = model.addVars(time_steps, lb=0, name="hydrogen_charge")
    hydrogen_discharge = model.addVars(time_steps, lb=0, name="hydrogen_discharge")
    conversion_factor = electrolyzer_eff * 3.6 / 120

    # Set objective function: minimize levelized hydrogen production cost
    wind_annual_cost = wind_cost * discount_rate / (1 - (1 + discount_rate) ** -wind_lifetime) + wind_om
    pv_annual_cost = pv_cost * discount_rate / (1 - (1 + discount_rate) ** -pv_lifetime) + pv_om
    battery_annual_cost = battery_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + battery_om
    electrolyzer_annual_cost = electrolyzer_cost * discount_rate / (1 - (1 + discount_rate) ** -electrolyzer_lifetime) + electrolyzer_om
    hydrogen_storage_annual_cost = hydrogen_storage_cost * discount_rate / (1 - (1 + discount_rate) ** -hydrogen_storage_lifetime) + hydrogen_storage_om

    model.setObjective(
        (wind_annual_cost * wind_capacity + 
         pv_annual_cost * pv_capacity +
         battery_annual_cost * battery_capacity +
           electrolyzer_annual_cost * electrolyzer_capacity + 
           hydrogen_storage_annual_cost * hydrogen_storage_capacity) +
        + surplus_penalty * sum(surplus[t] for t in range(time_steps)) / 8760
        + battery_charge_penalty * sum(battery_charge[t] for t in range(time_steps))
        + battery_discharge_penalty * sum(battery_discharge[t] for t in range(time_steps)),
        GRB.MINIMIZE
    )

    # Add constraints

    # Add constraint for electrolyzer power change rate (cannot exceed Flex per time step, except for the first time step)
    # for t in range(1, time_steps):
    #     model.addConstr(electrolyzer_power[t]+1 <= (1+Flex) * (electrolyzer_power[t-1]+1), name=f"electrolyzer_power_increase_limit_{t}")
    #     model.addConstr(electrolyzer_power[t]+1 >= (1-Flex) * (electrolyzer_power[t-1]+1), name=f"electrolyzer_power_decrease_limit_{t}")
    for t in range(1, time_steps):
        model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] <= Flex_down * electrolyzer_capacity,
                        name=f"electrolyzer_power_increase_limit_{t}")
        model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] >= -Flex_down * electrolyzer_capacity,
                        name=f"electrolyzer_power_decrease_limit_{t}")

    for t in range(time_steps):
        wind_power = wind_output[t] * wind_capacity
        pv_power = pv_output[t] * pv_capacity
        # Hydrogen storage balance constraint
        model.addConstr(electrolyzer_power[t] * conversion_factor + hydrogen_discharge[t] >= hydrogen_charge[t],
            name=f"hydrogen_min_balance_{t}")
        
        # Energy storage constraint, battery energy cannot exceed battery capacity
        model.addConstr(energy_balance[t] <= battery_capacity * battery_hours, name=f"energy_balance_limit_{t}")

        # Battery charging and discharging combined should be less than total installed capacity
        model.addConstr(battery_charge[t] + battery_discharge[t] <= battery_capacity, name=f"battery_charge_discharge_limit_{t}")

        # Power balance constraint (allow unmet demand or surplus)
        model.addConstr(
            wind_power + pv_power + battery_discharge[t] == electrolyzer_power[t] + battery_charge[t] + hydrogen_charge[t] * hydrogen_storage_elec + surplus[t],
            name=f"power_balance_{t}"
        )

        # Energy storage constraint, battery energy cannot exceed battery capacity
        if t == 0:
            model.addConstr(
                energy_balance[t] == energy_balance[8759],            # Initialize energy balance at the end of the year
                name=f"energy_balance_{t}"
            )
        else:
            model.addConstr(
                energy_balance[t] == energy_balance[t-1] + battery_charge[t] * battery_eff - battery_discharge[t] / battery_eff,
                name=f"energy_balance_{t}"
            )

    # Add constraint for annual hydrogen production
    annual_hydrogen_production = sum(electrolyzer_power[t] * conversion_factor for t in range(time_steps)) 
    model.addConstr(annual_hydrogen_production == target_hydrogen_production, name="annual_hydrogen_production")

    # Add constraint for wind and PV capacity sum being greater than zero
    model.addConstrs((electrolyzer_capacity >= electrolyzer_power[t] for t in range(time_steps)), name="electrolyzer_capacity_max_constraint")
    model.addConstr(wind_capacity + pv_capacity >= 0.1, name="wind_pv_capacity_sum")

    # Solve model
    model.setParam('Method', 2)  # Focus on finding a feasible solution
    model.setParam('Crossover', 0)

    model.optimize()

    # Store results
    if model.status == GRB.OPTIMAL:
        wind_capacity_val = wind_capacity.x
        pv_capacity_val = pv_capacity.x
        battery_capacity_val = battery_capacity.x
        electrolyzer_capacity_val = electrolyzer_capacity.x
        hydrogen_cost = model.objVal / target_hydrogen_production

        return {
            'wind_capacity': wind_capacity_val,
            'pv_capacity': pv_capacity_val,
            'battery_capacity': battery_capacity_val,
            'electrolyzer_capacity': electrolyzer_capacity_val,
            'hydrogen_cost': hydrogen_cost,
            'wind_cost': wind_cost,
            'pv_cost': pv_cost,
            'battery_cost': battery_cost,
            'electrolyzer_cost': electrolyzer_cost,
            'Flex': Flex_down
        }
    else:
        return {
            'wind_capacity': 0,
            'pv_capacity': 0,
            'battery_capacity': 0,
            'electrolyzer_capacity': 0,
            'hydrogen_cost': 0,
            'wind_cost': wind_cost,
            'pv_cost': pv_cost,
            'battery_cost': battery_cost,
            'electrolyzer_cost': electrolyzer_cost,
            'Flex': Flex_down
        }  # Return None if no feasible solution is found