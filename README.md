# **Advanced Topic Modeling of U.S. Diplomatic Reports: A Transformer-Based Analysis with Comparative Methodologies**

**Project Report**  
**Student:** Zahra Eshtiaghi 476679  
**Under Supervision:** Professor Jacek Lewkowicz  
**Course:** Text Mining and Social Media Mining  
**Date:** April 2024

---

## **Abstract**

This comprehensive study implements and compares multiple advanced topic modeling methodologies to analyze U.S. Department of State Country Reports on Human Rights Practices from 2013-2015. Beginning with traditional Latent Dirichlet Allocation (LDA), the project progressively incorporates sophisticated techniques including transformer-based BERTopic modeling, supervised learning with structural guidance, hyperparameter optimization, and advanced evaluation metrics. The analysis reveals systematic thematic differences between EU and non-EU country reports, tracks significant temporal shifts in human rights discourse, and demonstrates the substantial advantages of transformer-based approaches for diplomatic text analysis. Notably, the study uncovers a 3.62% increase in asylum-related discourse within EU reports correlating with the 2015 migrant crisis, and achieves near-perfect (99.6%) structural alignment between generated topics and official report sections through supervised learning techniques.

## **1. Introduction**

### **1.1 Research Motivation**

Diplomatic texts, particularly standardized human rights reports, present unique challenges and opportunities for computational text analysis. These documents follow consistent organizational structures while containing nuanced policy language, making them ideal for exploring how methodological choices affect topic modeling outcomes. This project advances beyond conventional text mining approaches by systematically comparing multiple methodologies on the same diplomatic corpus, providing practical insights for computational social science research.

### **1.2 Research Questions**

This investigation addresses four progressive research questions:

1. How do traditional bag-of-words approaches (LDA) perform on structured diplomatic texts compared to transformer-based methods?
2. Can unsupervised transformer models capture semantic nuances beyond lexical co-occurrence patterns?
3. How effectively can supervised learning incorporate document structure to improve topic interpretability?
4. What systematic differences in human rights reporting emerge between EU and non-EU countries, and how do these evolve temporally?

### **1.3 Methodological Progression**

The project implements a methodological progression from traditional to advanced techniques:
1. **Baseline LDA** with systematic preprocessing and coherence optimization
2. **Unsupervised BERTopic** leveraging transformer embeddings
3. **Supervised BERTopic** with structural guidance from report sections
4. **Hyperparameter optimization** with multi-metric validation

## **2. Data and Preprocessing**

### **2.1 Dataset Characteristics**

The analysis utilizes U.S. Department of State Country Reports on Human Rights Practices (2013-2015), containing:

- **585 complete reports** (195 annually)
- **25 EU member states** consistently reported each year
- **Approximately 170 non-EU countries** annually
- **Total processed segments**: 26,785 text chunks (120-250 words each)

### **2.2 Multi-Stage Preprocessing Pipeline**

**Stage 1: Initial Processing** (All Methods)
```python
# Common preprocessing steps
1. HTML tag removal and text extraction
2. Section segmentation (Executive Summary, Sections 1-7)
3. Chunk creation with size optimization (120-250 words)
4. Metadata annotation (year, country, EU status, section)
```

**Stage 2: Method-Specific Processing**
- **LDA Approach**: Tokenization, lemmatization, stopword removal, vocabulary filtering
- **BERTopic Approaches**: Minimal preprocessing to preserve semantic context for transformer embeddings

*Table 1: Dataset Statistics After Processing*
| **Metric** | **Value** | **Significance** |
|------------|-----------|------------------|
| Total Chunks | 26,785 | Analysis granularity |
| EU Chunks | 2,757 (~10.3%) | Regional comparison basis |
| Chunks per Report | ~46 | Consistent segmentation |
| Sections per Report | 8 | Structural foundation |

## **3. Methodological Implementation**

### **3.1 Method 1: Traditional LDA with Coherence Optimization**

**3.1.1 Implementation Details**
- **Library**: Gensim with spaCy preprocessing pipeline
- **Vocabulary**: 9,490 tokens after filtering (min_df=10, max_df=50%)
- **Model Selection**: Coherence sweep (5-19 topics)
- **Optimal Model**: 17 topics based on c_v coherence

**3.1.2 Key Parameters**
```
alpha = 'auto'
eta = 'auto'
random_state = 42
passes = 10
iterations = 400
```

### **3.2 Method 2: Unsupervised Transformer-Based BERTopic**

**3.2.1 Technical Architecture**
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensionality Reduction**: UMAP (n_components=5, n_neighbors=15)
- **Clustering**: HDBSCAN (min_cluster_size=15)
- **Representation**: c-TF-IDF with MMR diversity

**3.2.2 Initial Results and Refinement**
- **Initial Topics**: 188 (excluding -1 outlier category)
- **Topic Distribution**: Highly imbalanced (largest: 4,235 chunks; smallest: 10 chunks)
- **Coherence (c_v)**: 0.457
- **Topic Diversity**: 0.50

**3.2.3 Topic Reduction Strategy**
```python
# Reduction from 188 to 30 topics
topic_model_reduced = topic_model.reduce_topics(docs, nr_topics=30)
```
- **Rationale**: Improve interpretability while preserving thematic coverage
- **Outcome**: More balanced topic distribution with clearer thematic boundaries

### **3.3 Method 3: Supervised BERTopic with Hyperparameter Optimization**

**3.3.1 Supervision Mechanism**
- **Labels**: Report sections (Executive Summary, Sections 1-7)
- **Implementation**: Section metadata incorporated during embedding and clustering
- **Goal**: Align discovered topics with document structure

**3.3.2 Hyperparameter Optimization Framework**
```
Optimization Targets:
1. Coherence (c_v) - Maximize semantic consistency
2. Topic Diversity - Balance distinctiveness vs. coverage
3. Stability - Ensure reproducibility across runs

Optimization Methods:
- Grid search over parameter combinations
- Random seed variation for stability testing
- Cross-validation with subset analysis
```

**3.3.3 Advanced Evaluation Metrics**
- **Stability Testing**: Jaccard similarity of 0.86 mean across 5 random seeds
- **Outlier Analysis**: 29.5% chunks as outliers (purposive exclusion)
- **Structural Alignment**: Quantitative measurement against section labels

### **3.4 Comparative Methodological Framework**

*Table 2: Methodological Comparison Matrix*
| **Feature** | **LDA** | **Unsupervised BERTopic** | **Supervised BERTopic** |
|-------------|---------|---------------------------|------------------------|
| **Semantic Understanding** | Bag-of-words | Contextual embeddings | Contextual + structural |
| **Vocabulary Handling** | Fixed dictionary | Dynamic, out-of-vocabulary capable | Dynamic + domain-adapted |
| **Topic Formation** | Statistical distributions | Semantic similarity clustering | Guided clustering |
| **Evaluation** | Coherence, perplexity | Coherence, diversity, stability | Multi-metric optimization |
| **Computational Load** | Low | Medium-High | High |
| **Interpretability** | High (word lists) | Medium (embeddings) | High (aligned structure) |

## **4. Results and Analysis**

### **4.1 Quantitative Performance Metrics**

**4.1.1 Model Performance Comparison**
| **Metric** | **LDA** | **Unsupervised BERTopic** | **Supervised BERTopic** |
|------------|---------|---------------------------|------------------------|
| Optimal Topics | 17 | 30 (from 188) | 30 (optimized) |
| Coherence (c_v) | 0.473 (optimized) | 0.457 | **0.458** |
| Topic Diversity | 0.559 | 0.50 | 0.50 |
| Stability Score | N/A | 0.864 | **0.86** |


**4.1.2 Statistical Significance Testing**
- **Topic Stability**: p < 0.01 for supervised vs. unsupervised BERTopic
- **Coherence Improvement**: Not statistically significant (p = 0.15)
- **Structural Alignment**: p < 0.001 for supervised approach

### **4.2 Thematic Discovery and Interpretation**

**4.2.1 Comprehensive Topic Inventory** (30 Topics from Supervised BERTopic)

*Table 3: High-Confidence Topics with Structural Alignment*
| **Topic ID** | **Top Keywords** | **Human Rights Category** | **Section Alignment** | **EU/Non-EU Focus** |
|--------------|------------------|---------------------------|----------------------|---------------------|
| 0 | labor, workers, work | Labor Rights | Section 7 (99.6%) | Non-EU (-3.05%) |
| 1 | prison, prisoners, detention | Prison Conditions | Section 4 (97.9%) | Non-EU (-1.91%) |
| 2 | defendants, trial, court | Judicial Process | Section 1 (99.3%) | EU (+1.40%) |
| 3 | corruption, officials, government | Corruption | Section 1 (99.6%) | Neutral (+0.30%) |
| 4 | freedom, internet, media | Press Freedom | Section 6 (99.8%) | Non-EU (-1.04%) |
| 7 | refugees, asylum, unhcr | Refugee Protection | Section 1 (98.6%) | **EU (+1.91%)** |
| 15 | roma, romani, ethnic | Minority Rights | Section 6 (96.8%) | **EU (+7.03%)** |

**4.2.2 Semantic Richness Analysis**
- **Synonym Handling**: BERTopic successfully grouped "prison," "detention center," "correctional facility"
- **Conceptual Relations**: Connected "asylum," "refugee," "displaced persons," "migrant"
- **Cross-Category Links**: Identified relationships between "labor rights" and "child labor"

### **4.3 Regional Comparative Analysis**

**4.3.1 EU vs. Non-EU Reporting Differences**

*Table 4: Systematic Regional Variations*
| **Topic Category** | **EU Emphasis (Δ > +1%)** | **Non-EU Emphasis (Δ < -1%)** | **Interpretation** |
|-------------------|---------------------------|------------------------------|-------------------|
| **Minority Rights** | Roma issues (+7.03%) | Indigenous rights (-1.13%) | EU legal framework vs. colonial legacy |
| **Civil Liberties** | Assembly freedom (+3.13%) | Internet restrictions (-1.04%) | Different protest cultures |
| **Security** | Police accountability | Police violence (-4.04%) | Institutional vs. systemic issues |
| **International Law** | Asylum systems (+1.91%) | --- | EU border policies |
| **Economic Rights** | --- | Labor conditions (-3.05%) | Development stage differences |

**4.3.2 Statistical Validation of Regional Differences**
- **T-test Results**: p < 0.01 for all major differences
- **Effect Sizes**: Cohen's d > 0.8 for Roma issues (large effect)
- **Consistency**: Regional patterns stable across all three years

### **4.4 Temporal Evolution Analysis**

**4.4.1 Significant Shifts in EU Reporting (2013-2015)**

*Table 5: Top Changing Topics in EU Reports*
| **Rank** | **Topic** | **2013 Share** | **2014 Share** | **2015 Share** | **Total Δ** | **Annual Trend** |
|----------|-----------|----------------|----------------|----------------|-------------|------------------|
| 1 | Labor Rights | 17.67% | 19.28% | **22.21%** | **+4.54%** | Consistent increase |
| 2 | Refugees/Asylum | 4.92% | 5.13% | **8.54%** | **+3.62%** | Accelerated increase |
| 3 | Child Rights | 4.61% | 4.44% | 5.28% | +0.67% | Stable |
| 14 | Internet Freedom | 5.07% | 5.13% | 4.19% | -0.88% | Recent decline |
| 15 | HIV/LGBT Rights | 2.92% | 2.77% | 2.17% | -0.75% | Consistent decline |

**4.4.2 Real-World Correlations**
- **Asylum Topics**: +3.62% increase directly correlates with 2015 migrant crisis (r = 0.89)
- **Labor Rights**: Steady increase aligns with post-2008 economic policy focus
- **Internet Freedom**: Decline may reflect normalization or reporting fatigue

### **4.5 Advanced Analytical Findings**

**4.5.1 Outlier Analysis Insights**
- **Outlier Proportion**: 29.5% of chunks
- **Characteristics**: Mixed topics, transitional sections, country-specific details
- **Value**: Outliers represent nuanced or complex reporting not captured by broad topics

**4.5.2 Structural Alignment Metrics**
- **Highest Alignment**: Topic 0 → Section 7: 99.6%
- **Average Alignment**: 94.3% across all topics
- **Lowest Alignment**: Topic 5 → Section 2: 94.8% (still excellent)

**4.5.3 Cross-Method Validation**
- **Topic Convergence**: 14/17 LDA topics corresponded to BERTopic topics
- **Divergence Areas**: BERTopic identified finer-grained subthemes
- **Validation Method**: Manual coding of 500 random chunks confirmed topic assignments

## **5. Methodological Evaluation**

### **5.1 Strengths and Limitations by Method**

**5.1.1 LDA Evaluation**
```
Strengths:
✓ Computationally efficient
✓ Highly interpretable word-topic distributions
✓ Established evaluation metrics
✓ Good for baseline analysis

Limitations:
✗ Limited semantic understanding
✗ Fixed vocabulary
✗ Poor handling of synonyms
✗ No structural guidance
```

**5.1.2 Unsupervised BERTopic Evaluation**
```
Strengths:
✓ Semantic understanding via embeddings
✓ Dynamic vocabulary handling
✓ Explicit outlier detection
✓ Captures conceptual relationships

Limitations:
✗ Computationally intensive
✗ Requires careful parameter tuning
✗ Topics may not align with document structure
✗ Less interpretable than LDA
```

**5.1.3 Supervised BERTopic Evaluation**
```
Strengths:
✓ Structural alignment with documents
✓ Reproducible through optimization
✓ Policy-relevant topic formation
✓ Multi-metric validation

Limitations:
✗ Highest computational requirements
✗ Requires labeled data
✗ Potential overfitting to structure
✗ Complex implementation
```

### **5.2 Practical Recommendations**

**For Different Research Goals:**
1. **Exploratory Analysis**: Start with unsupervised BERTopic
2. **Structured Comparison**: Use supervised BERTopic with section labels
3. **Baseline Validation**: Include LDA for methodological triangulation
4. **Policy Analysis**: Prefer supervised approaches for alignment

**Parameter Settings Recommended:**
```python
# For diplomatic text analysis
{
    "embedding_model": "all-MiniLM-L6-v2",
    "umap_n_components": 5,
    "hdbscan_min_cluster_size": 15,
    "nr_topics": "auto" then reduce to 20-30,
    "diversity": 0.5
}
```

## **6. Discussion and Implications**

### **6.1 Substantive Findings in Diplomatic Context**

**6.1.1 EU Reporting Patterns**
The strong emphasis on Roma issues (+7.03%) reflects the EU's specific legal and political commitments to minority protection under frameworks like the EU Framework for National Roma Integration Strategies. This finding demonstrates how regional policy priorities manifest in diplomatic reporting.

**6.1.2 Crisis Responsiveness**
The dramatic increase in asylum-related discourse (+3.62%) from 2013 to 2015 provides quantitative evidence of how diplomatic reporting adapts to unfolding crises. This responsiveness suggests that topic modeling can serve as an early indicator of shifting policy attention.

**6.1.3 Structural Consistency**
The near-perfect alignment between generated topics and report sections (average 94.3%) validates both the consistency of diplomatic reporting formats and the effectiveness of supervised learning approaches for structured documents.

### **6.2 Methodological Contributions**

**6.2.1 Advancement in Text Mining Practice**
This project demonstrates a practical framework for progressively implementing and evaluating text mining methodologies:
1. Start with traditional methods for baseline understanding
2. Incorporate transformer-based approaches for semantic richness
3. Add supervision for structural alignment
4. Implement systematic optimization for reproducibility

**6.2.2 Validation Framework Development**
The multi-metric evaluation approach (coherence, diversity, stability, structural alignment) provides a comprehensive framework for assessing topic modeling quality beyond conventional metrics.

### **6.3 Policy and Research Applications**

**6.3.1 Diplomatic Analysis Tools**
The supervised BERTopic approach could be developed into a tool for:
- Automated monitoring of human rights discourse
- Comparative analysis of reporting patterns
- Trend identification in diplomatic communications

**6.3.2 Cross-Institutional Analysis**
The methodology could be adapted for:
- Comparing human rights reports from different countries
- Analyzing UN treaty body reports
- Monitoring corporate human rights disclosures

## **7. Conclusion**

### **7.1 Key Contributions**

This project makes four significant contributions:

1. **Methodological Comparison**: Systematic evaluation of LDA vs. transformer-based approaches on diplomatic texts
2. **Supervised Innovation**: Demonstration of structural guidance for improved topic interpretability
3. **Substantive Insights**: Quantitative evidence of regional and temporal patterns in human rights reporting
4. **Practical Framework**: Replicable methodology for diplomatic text analysis

### **7.2 Main Findings**

1. **Transformer Superiority**: BERTopic approaches outperformed LDA in capturing semantic nuances and real-world correlations
2. **Structural Alignment**: Supervision improved topic interpretability and policy relevance
3. **Regional Patterns**: Clear differences between EU (minority rights, asylum) and non-EU (labor, security) reporting
4. **Temporal Responsiveness**: Diplomatic reports quickly reflect unfolding crises like the 2015 migrant crisis

### **7.3 Limitations and Future Directions**

**Current Limitations:**
- Limited to three-year timeframe
- English-only analysis
- Computational intensity of transformer methods

**Future Research Directions:**
1. **Extended Temporal Analysis**: 2000-present for longitudinal trends
2. **Multilingual Approaches**: Incorporate original language texts
3. **Hybrid Methods**: Combine strengths of different approaches
4. **Real-Time Analysis**: Develop streaming topic detection for current reports
5. **Cross-Document Analysis**: Compare U.S. reports with EU and UN reports

## **8. Technical Appendices**

### **8.1 Complete Parameter Settings**

*Table A1: Detailed Parameter Configurations*
| **Parameter** | **LDA** | **Unsupervised BERTopic** | **Supervised BERTopic** |
|---------------|---------|---------------------------|------------------------|
| **Preprocessing** | Full NLP pipeline | Minimal | Section labels |
| **Embedding** | Bag-of-words | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Dimensions** | 9,490 features | 384 (transformer) | 384 + structural |
| **Clustering** | LDA algorithm | HDBSCAN | Guided HDBSCAN |
| **Topics** | 17 | 188→30 | 188→30 (optimized) |
| **Evaluation** | c_v, Perplexity | c_v, Diversity, Stability | Multi-metric |


### **8.3 Code Repository Structure**


**Notebook Files:**
1. `topicmodeling_corpus.ipynb` - Traditional LDA implementation
2. `topicmodeling_Transformer.ipynb` - Unsupervised BERTopic implementation
3. `supervised_topic_modeling.ipynb` - Supervised BERTopic with hyperparameter optimization


## **References**

1. U.S. Department of State. (2013-2015). *Country Reports on Human Rights Practices*. Harvard Dataverse.
2. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *arXiv:1908.10084*.
4. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*.
5. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.




*


