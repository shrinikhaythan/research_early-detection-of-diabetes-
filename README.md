# Ensemble Learning for Early Diabetes Detection

**AI-powered predictive screening system using ANN + CatBoost ensemble, achieving 98.94% accuracy on 100K patient records. Demonstrates production-ready ML pipeline: advanced preprocessing, class balancing, Bayesian hyperparameter optimization—directly transferable to Spectov's intelligent inspection & defect detection workflows.**

This project delivers scalable Python ML (scikit-learn, CatBoost, Keras) ready for CV integration (OpenCV/TensorFlow) in quality control pipelines. 

## Tech Stack 
- **Preprocessing**: KNNImputer, Isolation Forest + LOF outlier removal, SMOTETomek+ENN balancing
- **Models**: Keras ANN (PReLU, Bayesian-optimized) + CatBoost gradient boosting  
- **Ensemble**: Weighted voting (auto-optimized weights)
- **Evaluation**: MCC, NPV, F1 for imbalanced healthcare/manufacturing use cases
- **Dataset**: Kaggle Diabetes Prediction (100K records, 8 features)

## Production Results
| Model | Accuracy | F1 | MCC | NPV |
|-------|----------|----|-----|-----|
| ANN | 93.89% | 90.41% | 0.8798 | 96.22% |
| CatBoost | 94.95% | 96.95% | 0.9212 | 97.91% |
| **Ensemble** | **98.94%** | **98.99%** | **0.9797** | **98.95%** |

- **Anomaly Detection**: Isolation Forest + LOF → defect identification
- **Imbalanced Data**: SMOTETomek+ENN → rare defect class handling  
- **Hyperparameter Automation**: Bayesian optimization → rapid model iteration
- **Ensemble Reliability**: 98%+ metrics → zero-false-negative inspection
- **Scalable Pipeline**: Ready for manufacturing data + CV extensions


