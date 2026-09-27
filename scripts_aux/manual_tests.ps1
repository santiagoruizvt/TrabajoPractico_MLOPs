# Request ejemplo para un evento Bajo

Invoke-RestMethod -Method Post -Uri 'http://localhost:8800/predict' -ContentType 'application/json' -Body '{"civilian_ratio": 0.0, "country_te": 0.05, "date_prec": 1, "dyad_freq": 3, "dyad_id": 1042, "era_1989-99": 1, "era_2000-09": 0, "event_clarity": 1, "latitude": -8.5, "longitude": 27.3, "month_cos": 0.97, "month_sin": 0.26, "region_Africa": 1, "region_Americas": 0, "region_Asia": 0, "region_Europe": 0, "region_Middle East": 0, "tov_1": 0, "tov_2": 0, "tov_3": 1, "where_prec": 5, "year": 1993}'

# Request ejemplo para un evento Medio (pero devuelve Alto por la cercanía de clases)

Invoke-RestMethod -Method Post -Uri 'http://localhost:8800/predict' -ContentType 'application/json' -Body '{"civilian_ratio": 0.1, "country_te": 0.25, "date_prec": 3, "dyad_freq": 20, "dyad_id": 3301, "era_1989-99": 0, "era_2000-09": 1, "event_clarity": 1, "latitude": 6.3, "longitude": 20.1, "month_cos": 0.71, "month_sin": 0.71, "region_Africa": 1, "region_Americas": 0, "region_Asia": 0, "region_Europe": 0, "region_Middle East": 0, "tov_1": 0, "tov_2": 1, "tov_3": 0, "where_prec": 3, "year": 2007}'

# Request ejemplo para un evento Alto

Invoke-RestMethod -Method Post -Uri 'http://localhost:8800/predict' -ContentType 'application/json' -Body '{"civilian_ratio": 0.85, "country_te": 0.82, "date_prec": 1, "dyad_freq": 310, "dyad_id": 789, "era_1989-99": 0, "era_2000-09": 0, "event_clarity": 1, "latitude": 33.5, "longitude": 44.4, "month_cos": -0.5, "month_sin": 0.87, "region_Africa": 0, "region_Americas": 0, "region_Asia": 0, "region_Europe": 0, "region_Middle East": 1, "tov_1": 1, "tov_2": 0, "tov_3": 0, "where_prec": 1, "year": 2016}'