# Requests de ejemplo para la API (evento crudo UCDP-GED)

# Evento Medio Oriente, violencia estatal
Invoke-RestMethod -Method Post -Uri 'http://localhost:8800/predict' -ContentType 'application/json' -Body '{"country_id": 652, "dyad_new_id": 123, "date_start": "2015-03-14", "latitude": 33.5, "longitude": 36.3, "region": "Middle East", "type_of_violence": 1, "where_prec": 1, "date_prec": 1, "event_clarity": 1, "best": 10, "deaths_civilians": 5}'

# Evento en Africa, violencia unilateral, sin datos de muertes
Invoke-RestMethod -Method Post -Uri 'http://localhost:8800/predict' -ContentType 'application/json' -Body '{"country_id": 490, "dyad_new_id": 1042, "date_start": "1993-06-02", "latitude": -8.5, "longitude": 27.3, "region": "Africa", "type_of_violence": 3, "where_prec": 5, "date_prec": 1, "event_clarity": 1}'
