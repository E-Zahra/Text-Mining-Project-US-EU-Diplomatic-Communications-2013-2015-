# **Comparative Analysis of U.S. Department of State Human Rights Reports (2013-2015): A Multi-Method Topic Modeling Study**

**Group Project Report**  
**Team Members:** Zahra Eshtiaghi 476679, Si Tang Lin 476912, Dilara Ozdil 474544  
**Under Supervision:** Professor Jacek Lewkowicz  
**Course:** Text Mining and Social Media Mining  
**Date:** January 2026

---

## **Abstract**

This study employs multiple topic modeling approaches to analyze U.S. Department of State Country Reports on Human Rights Practices from 2013-2015. We implement three distinct methodological frameworks: (1) traditional Latent Dirichlet Allocation (LDA) with systematic preprocessing and quantitative evaluation, (2) unsupervised transformer-based BERTopic modeling, and (3) supervised BERTopic with hyperparameter optimization and structural guidance. Our comparative analysis reveals systematic differences in human rights reporting between EU and non-EU countries, tracks thematic evolution over the three-year period, and demonstrates how methodological choices influence topic interpretability and alignment with diplomatic document structures. The multi-method approach provides robust insights into both the content of diplomatic communications and the practical considerations for computational text analysis of structured governmental documents.

## **1. Introduction**

### **1.1 Research Context**

Diplomatic communications, particularly standardized human rights reports, represent rich textual data for understanding how states frame and prioritize international human rights issues. The U.S. Department of State's annual Country Reports on Human Rights Practices provide systematic documentation of human rights conditions worldwide, following a consistent structure across countries and years. These reports offer a valuable corpus for computational analysis of diplomatic discourse, regional reporting patterns, and temporal shifts in human rights priorities.

### **1.2 Research Questions**

This project addresses four primary research questions:

1. What are the dominant human rights themes in U.S. Department of State reports from 2013-2015?
2. How do thematic emphases differ between EU and non-EU country reports?
3. How did topic prevalence within EU reports evolve from 2013 to 2015?
4. How do different topic modeling methodologies (traditional LDA, unsupervised BERTopic, and supervised BERTopic) compare in extracting interpretable, structured themes from diplomatic documents?

### **1.3 Project Structure**

Our team implemented parallel methodological approaches to enable comparative analysis. One subgroup applied traditional corpus-based methods using Latent Dirichlet Allocation, while another employed advanced transformer-based techniques with both unsupervised and supervised configurations. This multi-method design allows us to assess methodological trade-offs and triangulate substantive findings.

## **2. Data**

### **2.1 Dataset Description**

We analyzed U.S. Department of State Country Reports on Human Rights Practices from 2013-2015, obtained from Harvard Dataverse (File ID: 4280620). The complete dataset includes:

- **585 reports** (195 per year)
- **25 EU member states** per year (consistent across all three years)
- **Approximately 170 non-EU countries** per year (varies slightly by year)
- **Total text chunks after preprocessing:** 26,785 segments

### **2.2 Data Preprocessing**

All methodological approaches shared common preprocessing steps:

1. **Text Extraction and Cleaning**: HTML tags and formatting artifacts were removed from original documents
2. **Section Segmentation**: Reports were divided into standardized sections (Executive Summary, Sections 1-7) based on heading patterns
3. **Chunk Creation**: Each section was further divided into 120-250 word chunks to enable fine-grained thematic analysis
4. **Metadata Annotation**: Each chunk received metadata tags for year, country, EU status, and source section

*Table 1: Dataset Composition*
| **Component** | **Value** | **Notes** |
|---------------|-----------|-----------|
| Time Period | 2013-2015 | Annual reporting cycle |
| Total Reports | 585 | 195 per year |
| EU Countries per Year | 25 | Fixed membership (2013-2015) |
| Text Chunks | 26,785 | After segmentation |
| EU Chunk Proportion | ~10.3% | Reflects shorter EU reports |

## **3. Methodology I: Traditional LDA Approach**

### **3.1 Implementation Details**

The LDA subgroup implemented a systematic corpus-based approach using established NLP libraries:

**3.1.1 Preprocessing Pipeline**
- Tokenization and lemmatization using spaCy NLP pipeline
- Removal of stopwords, punctuation, and non-alphabetic tokens
- Vocabulary filtering: minimum token length = 3 characters
- Dictionary construction with Gensim, filtering extremely rare (document frequency < 10) and overly common terms (document frequency > 50%)

**3.1.2 Model Training**
- Multiple LDA models trained with topic numbers ranging from 5 to 19
- Fixed parameters across all models for comparability
- Topic coherence (c_v metric) used to select optimal model
- Final selection: 17-topic model based on coherence optimization

### **3.2 Evaluation Framework**

- **Topic Coherence (c_v)**: Measures semantic consistency within topics
- **Topic Diversity**: Calculates lexical overlap across topics (target: low overlap)
- **Dominant Topic Assignment**: Each document assigned to its highest probability topic

## **4. Methodology II: Unsupervised Transformer-Based BERTopic**

### **4.1 Implementation Details**

This approach implemented advanced transformer-based methodology without explicit supervision:

**4.1.1 Embedding Generation**
- Used `sentence-transformers/all-MiniLM-L6-v2` for document embeddings
- Captures semantic similarity beyond word co-occurrence

**4.1.2 Model Configuration**
- Initial model produced 188 topics (excluding outliers)
- Reduced to 30 topics for interpretability while preserving thematic coverage
- HDBSCAN clustering for topic formation

### **4.2 Evaluation Framework**

- **Coherence (c_v)**: 0.457 for initial model
- **Topic Diversity**: 0.50 (top 10 words across topics)
- **Stability Testing**: Jaccard similarity of 0.864 mean across multiple runs
- **Outlier Management**: ~29% of chunks classified as outliers

## **5. Methodology III: Supervised BERTopic with Hyperparameter Optimization**

### **5.1 Implementation Details**

This advanced approach incorporated supervised learning and systematic optimization:

**5.1.1 Supervised Guidance**
- Incorporated report sections (Executive Summary, Sections 1-7) as supervision labels
- Guided model to align topics with document structure
- Ensured discovered topics correspond directly to official human rights categories

**5.1.2 Hyperparameter Optimization**
- Systematic tuning of model parameters
- Validated against three core quantitative metrics:
  - Coherence (C_v): Semantic similarity between top topic words
  - Topic Diversity: Ensures distinct, non-overlapping topics
  - Stability: Verified by re-running with multiple random seeds

**5.1.3 Optimization Strategy**
- Balanced topic separation and semantic interpretability
- Reduced from 188 to 30 topics for optimal interpretability
- Maintained statistical robustness through rigorous validation

### **5.2 Evaluation Framework**

- **Quantitative Performance**:
  - Topic Coherence (C_v): 0.458
  - Topic Stability (Jaccard Overlap): 0.86 mean
  - Topic Diversity: 0.50
  
- **Structural Alignment Metrics**:
  - Labor Rights → Section 7 alignment: 99.6%
  - Prison Conditions → Section 1 alignment: High correspondence
  - Gender-Based Violence → Section 6 alignment: Consistent mapping

## **6. Results: Comparative Analysis**

### **6.1 Methodological Performance Comparison**

*Table 2: Methodological Performance Metrics*
| **Metric** | **LDA Approach** | **Unsupervised BERTopic** | **Supervised BERTopic** | **Interpretation** |
|------------|------------------|---------------------------|------------------------|-------------------|
| **Number of Topics** | 17 | 30 (from 188) | 30 (from 188) | BERTopic captures finer granularity |
| **Vocabulary Size** | 9,490 tokens | Full contextual embedding | Full contextual embedding | Transformers use semantic similarity |
| **Topic Diversity** | 0.735 | 0.50 | 0.50 | LDA produces more distinct topics |
| **Coherence (c_v)** | Optimized via sweep | 0.457 | 0.458 | Comparable coherence across methods |
| **Structural Alignment** | Indirect | Indirect | Direct (99.6%) | Supervision dramatically improves alignment |
| **Outlier Rate** | N/A | 29.5% | 29.5% | Consistent outlier handling |

### **6.2 Thematic Coverage: Common Findings Across Methods**

All three methods identified consistent human rights themes, though with different granularity and structural alignment:

**High-Confidence Themes (All Methods):**
1. Judicial processes and court systems
2. Prison conditions and detention
3. Labor rights and working conditions
4. Refugee and asylum systems
5. Gender-based violence
6. Freedom of expression and media

### **6.3 EU vs. Non-EU Comparison: Convergent Results**

*Table 3: Regional Differences in Reporting Emphasis*
| **Theme Category** | **More Emphasized in EU Reports** | **More Emphasized in Non-EU Reports** | **Delta (EU - Non-EU)** |
|-------------------|-----------------------------------|--------------------------------------|------------------------|
| **Minority Rights** | Roma/Romani Issues | Indigenous land rights | +7.03% |
| **Civil Liberties** | Freedom of Assembly | Internet restrictions | +3.13% |
| **Refugee Systems** | Asylum procedures | --- | +1.91% |
| **Security Issues** | Police accountability | Police violence, killings | -4.04% |
| **Labor Rights** | --- | Working conditions | -3.05% |
| **Detention** | --- | Prison conditions | -1.91% |

### **6.4 Temporal Trends: EU Reports 2013-2015**

*Table 4: Thematic Evolution in EU Reports*
| **Rank** | **Topic ID** | **Topic Label** | **2013 Share** | **2015 Share** | **Change** | **Real-World Correlation** |
|----------|--------------|-----------------|----------------|----------------|------------|---------------------------|
| 1 | 0 | Labor Rights | 17.67% | 22.21% | **+4.54%** | Economic focus post-crisis |
| 2 | 7 | Refugees/Asylum | 4.92% | 8.54% | **+3.62%** | 2015 European migrant crisis |
| 3 | 8 | Child Rights | 4.61% | 5.28% | +0.67% | Ongoing child protection |
| 13 | 2 | Judicial Process | 9.68% | 8.85% | -0.83% | Possible normalization |
| 14 | 4 | Internet Freedom | 5.07% | 4.19% | -0.88% | Variable reporting emphasis |
| 15 | 14 | HIV/LGBT Rights | 2.92% | 2.17% | -0.75% | Shifting priorities |

### **6.5 Supervised BERTopic: Structural Alignment Results**

The supervised approach demonstrated exceptional alignment with document structure:

1. **Labor Rights Topics**: 99.6% alignment with Section 7 (Worker Rights)
2. **Prison Conditions**: Primary mapping to Section 1 (Respect for Integrity of Person)
3. **Gender-Based Violence**: Consistent emergence from Section 6 (Discrimination)
4. **Refugee Topics**: Strong correspondence with asylum-related sections
5. **Judicial Processes**: Alignment with legal procedure sections

## **7. Method-Specific Findings**

### **7.1 LDA-Specific Insights**

**7.1.1 Topic Interpretability**
- The 17-topic model showed excellent topic separation (diversity: 0.735)
- Topics corresponded clearly to human rights categories
- Manual inspection confirmed face validity of topic-word distributions

**7.1.2 Statistical Patterns**
- Most prevalent topic: "court/case" (≈10.8% of chunks)
- Clear topic-document distributions enabled quantitative comparison
- Stable patterns across multiple model initializations

### **7.2 Unsupervised BERTopic Insights**

**7.2.1 Semantic Richness**
- Transformer embeddings captured nuanced thematic relationships
- Identified subtopics within broader categories
- Better handling of synonymy and related concepts

**7.2.2 Outlier Management**
- Explicit outlier detection (29.5% of chunks)
- Ensured final topics represented coherent themes
- Improved overall model quality

### **7.3 Supervised BERTopic: Advanced Insights**

**7.3.1 Optimization Benefits**
- Hyperparameter tuning balanced interpretability and statistical rigor
- Systematic validation against multiple metrics
- Reproducible results across random seeds

**7.3.2 Supervised Guidance Impact**
- Near-perfect structural alignment (99.6% for key topics)
- Topics directly correspond to diplomatic reporting categories
- Enhanced utility for policy analysis and comparative studies

## **8. Discussion**

### **8.1 Substantive Implications**

**8.1.1 Diplomatic Reporting Patterns**
Our analysis reveals systematic differences in how the U.S. Department of State frames human rights issues across regions. EU reports emphasize institutional protections and minority rights, reflecting the EU's legal framework and political priorities. Non-EU reports focus more on conflict, security, and basic labor rights, potentially reflecting different human rights challenges.

**8.1.2 Temporal Responsiveness**
The increase in asylum-related content within EU reports (2013-2015) demonstrates how diplomatic reporting responds to unfolding crises. The +3.62% increase in refugee/asylum topics directly correlates with the 2015 European migrant crisis, validating the model's sensitivity to real-world events.

**8.1.3 Structural Consistency**
The high structural alignment in supervised BERTopic confirms that diplomatic reports follow consistent organizational patterns, enabling reliable computational analysis.

### **8.2 Methodological Reflections**

**8.2.1 Trade-offs Between Approaches**
- **LDA Strengths**: Computational efficiency, interpretable word-topic distributions, established evaluation metrics
- **Unsupervised BERTopic Strengths**: Semantic understanding, handling of out-of-vocabulary terms, outlier management
- **Supervised BERTopic Strengths**: Structural alignment, reproducibility, policy relevance
- **Practical Considerations**: Each method offers different balances of interpretability, accuracy, and alignment

**8.2.2 Methodological Recommendations**
1. **Exploratory Analysis**: Use unsupervised BERTopic for initial theme discovery
2. **Structured Analysis**: Apply supervised BERTopic when document structure is known
3. **Comparative Studies**: Use LDA for cross-method validation
4. **Policy Applications**: Prefer supervised approaches for alignment with official categories

## **9. Conclusion**

This multi-method study demonstrates the value of computational text analysis for understanding diplomatic communications. By applying three distinct topic modeling approaches to U.S. Department of State human rights reports, we have:

1. **Identified Consistent Thematic Patterns**: All methods revealed clear regional differences and temporal trends
2. **Validated Methodological Approaches**: Convergence across methods strengthens confidence in results
3. **Demonstrated Advanced Techniques**: Supervised BERTopic with hyperparameter optimization offers superior structural alignment
4. **Provided Actionable Insights**: Regional and temporal patterns offer concrete understanding of diplomatic priorities
5. **Advanced Methodological Understanding**: Comparative analysis illuminates trade-offs between different text mining techniques

The project highlights how computational methods can complement traditional diplomatic analysis, providing systematic, scalable approaches to understanding how states frame and prioritize human rights issues in international discourse.

## **10. Limitations and Future Work**

### **10.1 Limitations**
- **Temporal Scope**: Limited to three years; longer time series would provide more robust trend analysis
- **Language Constraint**: Analysis limited to English-language reports
- **Contextual Nuance**: Topic models may miss subtle rhetorical strategies or diplomatic signaling
- **Evaluation Metrics**: Standard coherence metrics may not fully capture topic quality for specialized domains

### **10.2 Future Directions**
1. **Expand Temporal Range**: Analyze reports from 2000-present to capture longer-term trends
2. **Comparative Framework**: Apply same methods to other countries' human rights reports
3. **Integration with Event Data**: Correlate topic prevalence with real-world human rights events
4. **Advanced Visualization**: Develop interactive tools for exploring topic relationships
5. **Multimodal Analysis**: Incorporate visual elements from reports where available

## **11. References**

1. U.S. Department of State. (2013-2015). *Country Reports on Human Rights Practices*. Harvard Dataverse.
2. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*.
3. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
4. DiMaggio, P., Nag, M., & Blei, D. (2013). Exploiting affinities between topic modeling and the sociological perspective on culture. *Poetics*.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *arXiv preprint arXiv:1908.10084*.

## **Appendix: Technical Implementation Details**

Complete code, preprocessing scripts, model configurations, and analysis notebooks are available in our project repository:

**Notebook Files:**
1. `topicmodeling_corpus.ipynb` - Traditional LDA implementation
2. `topicmodeling_Transformer_Final.ipynb` - Unsupervised BERTopic implementation
3. `supervised_topic_modeling.ipynb` - Supervised BERTopic with hyperparameter optimization

*Table A1: Complete Parameter Settings*
| **Parameter** | **LDA Value** | **Unsupervised BERTopic** | **Supervised BERTopic** |
|---------------|---------------|---------------------------|------------------------|
| **Preprocessing** | spaCy | Minimal preprocessing | Section labels as supervision |
| **Embedding Model** | Bag-of-words | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Topics** | 17 (optimized) | 188 → 30 (reduced) | 188 → 30 (optimized) |
| **Coherence (c_v)** | Optimized via sweep | 0.457 | 0.458 |
| **Topic Diversity** | 0.735 | 0.50 | 0.50 |
| **Stability Testing** | Multiple seeds | Jaccard: 0.864 | Jaccard: 0.86 |
| **Structural Alignment** | Indirect | Indirect | 99.6% (Labor→Section 7) |

*Table A2: Key Performance Indicators*
| **KPI** | **LDA** | **Unsupervised BERTopic** | **Supervised BERTopic** | **Best Performer** |
|---------|---------|---------------------------|------------------------|-------------------|
| **Interpretability** | High | Medium | **High** | Supervised BERTopic |
| **Structural Alignment** | Low | Low | **High** | Supervised BERTopic |
| **Statistical Robustness** | High | Medium | **High** | Supervised BERTopic |
| **Computational Efficiency** | **High** | Medium | Low | LDA |
| **Real-World Correlation** | Medium | High | **High** | Supervised BERTopic |

---

**Course:** Text Mining and Social Media Mining  
**Instructor:** Professor Jacek Lewkowicz