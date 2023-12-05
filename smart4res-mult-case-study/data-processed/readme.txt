pv_power_clean: power production data from base-level PVs. Columns refers to Park-ID from metadata file
pv-metadata: park-id, latitude, longitude
pv_weather_pred: dictionary that maps park-id (key) to weather forecasts (item)
		 Right-join on index with pv_power_clean.csv
		 Weather forecasts from the closest grid point (based on Eucl. distance)
		 Issued at 00:00 D-1, with horizon 96-191 steps ahead (15-min)

* Everything is downscaled to hourly timesteps*

wind/pv_power_clean_imp: missing values imputation as the **average from available farms**
wind/pv_power_clean: missing values imputation with **linear interpolation**

wind/pv_meteo_id: dictionary that maps wind turbine/PV ID to closest NWP grid point
wind/pv_weather_pred: dictionary that maps wind turbine/PV ID to NWPs from the closest grid point 