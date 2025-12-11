# Citywide Demographic and Socioeconomic Analysis of Chicago's Community Areas Using Census & ACS Data

### Final Project Report

**Author:** Sai Shashank Satuluri\
**A#:** A20601123

## Introduction

This project analyzes how Chicago's neighborhoods --- Community Areas
(CAs) --- have evolved socially, economically, and demographically over
time. It uses Census 2010 & 2020, ACS 2023, and TIGER/Line geospatial
data. The project expanded from two neighborhoods to all 77 Community
Areas.

## Project Workflow

1.  Select community areas and block groups\
2.  Fetch Census 2010 & 2020\
3.  Compute demographic changes\
4.  Download ACS 2023\
5.  Aggregate block groups → community areas\
6.  Build feature vectors\
7.  Construct socioeconomic + spatial networks\
8.  Leiden clustering\
9.  Visualizations\
10. Key insights

## Data Sources

-   Decennial Census 2010 & 2020\
-   ACS 2023 5-year estimates\
-   TIGER/Line shapefiles\
-   Chicago Community Areas GeoJSON

## Scripts Overview

-   `01_ca_selection.py`\
-   `02_decinnial_analysis.py`\
-   `03_acs_download_and_features.py`\
-   `04_aggregate_acs_to_ca.py`\
-   `05_build_features_and_network.py`\
-   `06_decennial_key_findings.py`\
-   `07_viusalize_decinnial_data.py`\
-   `08_comparative_plots.py`\
-   `09_additional_visualizations.py`

## Results Summary

### 2010--2020 Census

-   White: 32% → 29%\
-   Black: 33% → 29%\
-   Asian: 6% → 7%\
-   Hispanic: 29% → 35%\
-   Diversity Index: 0.78 → 0.81

### ACS 2023

-   Median income: \$30K → \$120K\
-   Poverty: 5% → 40%\
-   Bachelor's degree: 20% → 80%

### Clustering

-   Leiden clustering reveals clear socioeconomic groups\
-   t-SNE shows high feature separability

### Combined Insights

-   Hispanic population growth correlated with unemployment (r ≈ 0.45)\
-   Population decline correlated with lower income (r ≈ --0.32)\
-   No Asian-majority neighborhoods

## Discussion

### Strengths

-   Efficient API-based pipeline\
-   Accurate geospatial processing\
-   Strong visualizations

### Challenges

-   TIGER/Line learning curve\
-   API rate limits\
-   Missing ACS values

### Future Work

-   Predictive modeling\
-   Multi-year ACS time-series\
-   Interactive dashboards\
-   Cloud deployment

## Execution

``` bash
python 01_ca_selection.py
python 02_decinnial_analysis.py
python 03_acs_download_and_features.py
python 04_aggregate_acs_to_ca.py
python 05_build_features_and_network.py
python 06_decennial_key_findings.py
python 07_viusalize_decinnial_data.py
python 08_comparative_plots.py
python 09_additional_visualizations.py
```

## References

-   U.S. Census Bureau\
-   TIGER/Line\
-   Chicago Data Portal\
-   Kyle Walker --- *Analyzing US Census Data*\
-   GeoPandas, Pandas, Scikit-learn, Leidenalg
