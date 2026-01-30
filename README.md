# **Comparative Analysis of U.S. Department of State Human Rights Reports (2013-2015): A Multi-Method Topic Modeling Study**

**Group Project Report**  
**Team Members:** Zahra Eshtiaghi 476679, Si Tang Lin 476912, Dilara Ozdil 474544
** Under Supervision : Ptofessor Jacek Lewkowicz 
**Course:** Text Mining and Social Media Mining


---

## **Abstract**

This study employs multiple topic modeling approaches to analyze U.S. Department of State Country Reports on Human Rights Practices from 2013-2015. We implement two distinct methodological frameworks: (1) traditional Latent Dirichlet Allocation (LDA) with systematic preprocessing and quantitative evaluation, and (2) transformer-based BERTopic with supervised learning and hyperparameter optimization. Our comparative analysis reveals systematic differences in human rights reporting between EU and non-EU countries, tracks thematic evolution over the three-year period, and demonstrates how methodological choices influence topic interpretability and alignment with diplomatic document structures. The multi-method approach provides robust insights into both the content of diplomatic communications and the practical considerations for computational text analysis of structured governmental documents.

## **1. Introduction**

### **1.1 Research Context**

Diplomatic communications, particularly standardized human rights reports, represent rich textual data for understanding how states frame and prioritize international human rights issues. The U.S. Department of State's annual Country Reports on Human Rights Practices provide systematic documentation of human rights conditions worldwide, following a consistent structure across countries and years. These reports offer a valuable corpus for computational analysis of diplomatic discourse, regional reporting patterns, and temporal shifts in human rights priorities.

### **1.2 Research Questions**

This project addresses four primary research questions:

1. What are the dominant human rights themes in U.S. Department of State reports from 2013-2015?
2. How do thematic emphases differ between EU and non-EU country reports?
3. How did topic prevalence within EU reports evolve from 2013 to 2015?
4. How do different topic modeling methodologies (traditional LDA vs. transformer-based BERTopic) compare in extracting interpretable, structured themes from diplomatic documents?

### **1.3 Project Structure**

Our team implemented parallel methodological approaches to enable comparative analysis. One subgroup applied traditional corpus-based methods using Latent Dirichlet Allocation, while another employed advanced transformer-based techniques with supervised guidance. This multi-method design allows us to assess methodological trade-offs and triangulate substantive findings.

## **2. Data**

### **2.1 Dataset Description**

We analyzed U.S. Department of State Country Reports on Human Rights Practices from 2013-2015, obtained from Harvard Dataverse (File ID: 4280620). The complete dataset includes:

- **585 reports** (195 per year)
- **25 EU member states** per year (consistent across all three years)
- **Approximately 170 non-EU countries** per year (varies slightly by year)
- **Total text chunks after preprocessing:** 26,785 segments

### **2.2 Data Preprocessing**

Both methodological approaches shared common preprocessing steps:

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

## **4. Methodology II: Transformer-Based BERTopic Approach**

### **4.1 Implementation Details**

The BERTopic subgroup implemented an advanced transformer-based methodology:

**4.1.1 Embedding Generation**
- Used `sentence-transformers/all-MiniLM-L6-v2` for document embeddings
- Captures semantic similarity beyond word co-occurrence

**4.1.2 Supervised Topic Modeling**
- Incorporated report sections as supervision labels
- Guided model to align topics with document structure
- Implemented extensive hyperparameter optimization

**4.1.3 Topic Reduction Strategy**
- Initial model produced 188 topics (excluding outliers)
- Reduced to 30 topics for interpretability while preserving thematic coverage

### **4.2 Evaluation Framework**

- **Coherence (c_v)**: 0.458 for final reduced model
- **Topic Diversity**: 0.50 (top 10 words across topics)
- **Stability Testing**: Jaccard similarity of 0.86 mean across multiple runs
- **Outlier Management**: ~29% of chunks classified as outliers

## **5. Results: Comparative Analysis**

### **5.1 Methodological Performance Comparison**

*Table 2: Methodological Performance Metrics*
| **Metric** | **LDA Approach** | **BERTopic Approach** | **Interpretation** |
|------------|------------------|----------------------|-------------------|
| **Number of Topics** | 17 | 30 (reduced from 188) | BERTopic captures finer granularity |
| **Vocabulary Size** | 9,490 tokens | Full contextual embedding | BERTopic uses semantic similarity |
| **Topic Diversity** | 0.735 | 0.50 | LDA produces more distinct topics |
| **Outlier Rate** | Not applicable | 29.5% | BERTopic explicitly handles outliers |
| **Structural Alignment** | Indirect | Direct (supervised) | BERTopic better aligns with document structure |

### **5.2 Thematic Coverage: Common Findings**

Both methods identified consistent human rights themes, though with different granularity and terminology:

**High-Confidence Themes (Both Methods):**
1. Judicial processes and court systems
2. Prison conditions and detention
3. Labor rights and working conditions
4. Refugee and asylum systems
5. Gender-based violence
6. Freedom of expression and media

### **5.3 EU vs. Non-EU Comparison: Convergent Results**

*Table 3: Regional Differences in Reporting Emphasis*
| **Theme Category** | **More Emphasized in EU Reports** | **More Emphasized in Non-EU Reports** | **Consistency Across Methods** |
|-------------------|-----------------------------------|--------------------------------------|-------------------------------|
| **Minority Rights** | Roma issues, anti-discrimination | Indigenous land rights | High |
| **Civil Liberties** | Freedom of assembly, expression | Internet restrictions | Medium |
| **Security Issues** | Police accountability | Police violence, killings | High |
| **Social Protections** | Disability rights | Labor conditions | Medium |
| **International Systems** | Asylum procedures, UNHCR | --- | High |

### **5.4 Temporal Trends: EU Reports 2013-2015**

*Table 4: Thematic Evolution in EU Reports*
| **Trend Direction** | **Specific Themes Increasing** | **Specific Themes Decreasing** | **Real-World Correlation** |
|---------------------|--------------------------------|--------------------------------|----------------------------|
| **Increasing** | Refugee/asylum topics (+3.6%) | --- | 2015 European migrant crisis |
| | Labor rights (+4.5%) | --- | Economic focus post-crisis |
| **Decreasing** | --- | Internet freedom (-0.9%) | Possible normalization |
| | --- | HIV/LGBT rights (-0.7%) | Variable reporting emphasis |

## **6. Method-Specific Findings**

### **6.1 LDA-Specific Insights**

**6.1.1 Topic Interpretability**
- The 17-topic model showed excellent topic separation (diversity: 0.735)
- Topics corresponded clearly to human rights categories
- Manual inspection confirmed face validity of topic-word distributions

**6.1.2 Statistical Patterns**
- Most prevalent topic: "court/case" (≈10.8% of chunks)
- Clear topic-document distributions enabled quantitative comparison
- Stable patterns across multiple model initializations

### **6.2 BERTopic-Specific Insights**

**6.2.1 Structural Alignment**
- Near-perfect alignment (99.6%) between "Labor Rights" topics and Section 7
- Strong correspondence between generated topics and report sections
- Supervision effectively guided topic formation

**6.2.2 Semantic Richness**
- Transformer embeddings captured nuanced thematic relationships
- Identified subtopics within broader categories (e.g., different aspects of prison conditions)
- Better handling of synonymy and related concepts

## **7. Discussion**

### **7.1 Substantive Implications**

**7.1.1 Diplomatic Reporting Patterns**
Our analysis reveals systematic differences in how the U.S. Department of State frames human rights issues across regions. EU reports emphasize institutional protections and minority rights, reflecting the EU's legal framework and political priorities. Non-EU reports focus more on conflict, security, and basic labor rights, potentially reflecting different human rights challenges.

**7.1.2 Temporal Responsiveness**
The increase in asylum-related content within EU reports (2013-2015) demonstrates how diplomatic reporting responds to unfolding crises. This finding validates the utility of topic modeling for tracking real-time shifts in diplomatic attention.

### **7.2 Methodological Reflections**

**7.2.1 Trade-offs Between Approaches**
- **LDA Strengths**: Computational efficiency, interpretable word-topic distributions, established evaluation metrics
- **BERTopic Strengths**: Semantic understanding, handling of out-of-vocabulary terms, integration of document structure
- **Practical Considerations**: LDA required more manual preprocessing but produced more immediately interpretable topics; BERTopic required less preprocessing but more parameter tuning

**7.2.2 Recommendations for Future Research**
1. **Hybrid Approaches**: Combine LDA's interpretability with BERTopic's semantic understanding
2. **Dynamic Topic Modeling**: Extend analysis to track topic evolution more continuously
3. **Multilingual Analysis**: Incorporate non-English source materials where available
4. **Cross-Validation**: Apply similar methods to other diplomatic corpora

## **8. Conclusion**

This multi-method study demonstrates the value of computational text analysis for understanding diplomatic communications. By applying both traditional LDA and advanced transformer-based approaches to U.S. Department of State human rights reports, we have:

1. **Identified Consistent Thematic Patterns**: Both methods revealed clear regional differences and temporal trends in human rights reporting
2. **Validated Methodological Approaches**: The convergence of findings across methods strengthens confidence in results
3. **Demonstrated Practical Utility**: The analysis provides actionable insights for policymakers and researchers
4. **Advanced Methodological Understanding**: Our comparative approach illuminates trade-offs between different text mining techniques

The project highlights how computational methods can complement traditional diplomatic analysis, providing systematic, scalable approaches to understanding how states frame and prioritize human rights issues in international discourse.

## **9. Limitations and Future Work**

### **9.1 Limitations**
- **Temporal Scope**: Limited to three years; longer time series would provide more robust trend analysis
- **Language Constraint**: Analysis limited to English-language reports
- **Contextual Nuance**: Topic models may miss subtle rhetorical strategies or diplomatic signaling
- **Evaluation Metrics**: Standard coherence metrics may not fully capture topic quality for specialized domains

### **9.2 Future Directions**
1. **Expand Temporal Range**: Analyze reports from 2000-present to capture longer-term trends
2. **Comparative Framework**: Apply same methods to other countries' human rights reports
3. **Integration with Event Data**: Correlate topic prevalence with real-world human rights events
4. **Advanced Visualization**: Develop interactive tools for exploring topic relationships

## **10. References**

1. U.S. Department of State. (2013-2015). *Country Reports on Human Rights Practices*. Harvard Dataverse.
2. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*.
3. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
4. DiMaggio, P., Nag, M., & Blei, D. (2013). Exploiting affinities between topic modeling and the sociological perspective on culture. *Poetics*.

## **Appendix: Technical Implementation Details**
topicmodeling_corpus.ipynb
topicmodeling_Transformer_Final.ipynb
supervised_topic_modeling.ipynb


*Table A1: Complete Parameter Settings*
| **Parameter** | **LDA Value** | **BERTopic Value** |
|---------------|---------------|-------------------|
| Preprocessing Library | spaCy | sentence-transformers |
| Embedding Model | Bag-of-words | all-MiniLM-L6-v2 |
| Topic Selection | Coherence sweep | Reduction from 188→30 |
| Evaluation Metrics | c_v, Diversity | c_v, Diversity, Stability |
| Random Seeds | Multiple | Multiple for stability |


