### PowerAnomalyDetection
**Overall explanation**
We detect anomalies of weekends and weekdays respectively in different models.
- For weekend anomalies, we used mainly Isolation Forest, and got very good performance on precision.
- For weekday anomalies, we used mainly XGBoost, and we have two versions of codes:
 	- `Code in R`: As one of our team member is more familiar with R, he used R to build models to detect weekday anomalies during the competition;
 	- `Code in python`: We rewrite the R version code in python, as we want to get consistent code style. But the results has some differences with the R version, we thus submit the two versions together here.

**Code explanation**
- Step1: `make_dataset.py` Transform the raw data to proper input data formats.
- Step2: `models_weekend.py`  Get weekend and holiday anomalies.
- Step3:
    - `models_weekday.py` Get weekday anomalies using python.
    - `models_weekday.R` Get weekday anomalies using R.
- Step4: `combine_results.py` Combine weekend and weekday anomalies.


**Result explanation**
- Result with best precision score:
    - `res_weekend_without_holiday.csv` Only contains anomalies of weekends.
    - `res_weekend_with_holiday.csv` Contains anomalies of weekends and holidays.

- Results of weekdays:
    - `res_weekday_r_version.csv` Contains anomalies of weekdays implemented in R, the version we used during the competition.
    - `res_weekday_p_version.csv` Contains anomalies of weekdays implemented in python, the version we rewrite after the competition.
- Combined results:
    - `res_combined_r.csv` Contains anomalies of both weekends, holidays and weekdays(R version)
    - `res_combined_p.csv` Contains anomalies of both weekends, holidays and weekdays(python version)
