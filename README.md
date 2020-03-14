[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata-public-assets/se-challenge-2-banner.jpg)

#  Power Laws: Anomaly Detection
## Goal of the Competition

Energy consumption of buildings has steadily increased. There is an increasing realization that many buildings do not perform as intended by their designers. Typical buildings consume 20% more energy than necessary due to faulty construction, malfunctioning equipment, incorrectly configured control systems and inappropriate operating procedures.

The building systems may fail to meet the performance expectations due to various faults. Poorly maintained, degraded, and improperly controlled equipment wastes an estimated 15% to 30% of energy used in commercial buildings.

Therefore, it is of great potential to develop automatic, quick-responding, accurate and reliable fault detection and to provide diagnosis schemes to ensure the optimal operations of systems to save energy.

** In [this competition](https://www.drivendata.org/competitions/52/anomaly-detection-electricity/) data scientists all over the world built algorithms to forecast building consumption reliably. **

Identifying anomalies was a tricky task, and the best performers combined human judgment with machine suggestions. Winners documented these approaches and sugggested new ways of thinking about anomalies in the context of energy usage. Ultimately, these ideas can help drive forward the kinds of anomalies that we hope to identify in building energy use.

## What's in this Repository
This repository contains code from winning competitors in the [Power Laws: Anomaly Detection](https://www.drivendata.org/competitions/51/electricity-prediction-machine-learning/) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | PINGANAI_ | 0.8073 | 0.8073   | We mainly used two algorithms to detect anomalies: Isolation Forest for weekends (including holidays) and XGBoost for weekdays.
2 | lviana | 0.8028 | 0.8027 | I combine prediction-based and rule-based approaches to detect abnormal energy consumption. The main idea is to fit a machine learning model that predicts the energy consumption for the next timestamp, then we measure kind of the level of “surprise” of the model based on the gap between the prediction and the true consumption. If we find a gap, then either (i) the model actually made a mistake, or (ii) an abnormal energy usage has happened. When detecting overconsumption, we want to avoid the cases where the model just made a mistake.
3 | viana | 0.8001 | 0.8000 | Hand picking and LB probing algorithm taking advantage of optimization metric. Huge penalty for FP answers convinced me to find several concrete examples of anomalies and to get public score around 80%. Other motivation for that was to gain an understanding of what is meant by anomaly. To find these examples I grouped-by observations with similar time related features and marked 4 sigma deviations as anomalies. Then by manually examining groups of anomalies I started LB probing for each meter_id separately. After a week or so I got the expected score.  That said, my approaches to finding initial candidates for anomalies might be useful.
Best Report | pavel_kuzman | 0.7111 | 0.6008 | Pavel created models for identifying anomalies using k-nearest neighbors and neiral networks. After this he explored the kinds of anomalies that were detected using these methods and how identification of anomalies could be improved with additional data.

#### [Interview with winners](http://drivendata.co/blog/power-laws-anomalies-winners/)
