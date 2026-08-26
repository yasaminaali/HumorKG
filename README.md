A knowledge-graph-grounded humor generation system for [SemEval-2026 Task 1: MWAHAHA](https://semeval.github.io/SemEval2026/) (English / Spanish / Chinese).                                          
   
Given a news headline (or a pair of words), HumorKG retrieves related concepts from a knowledge graph (WordNet or ConceptNet) in the target language, appends them to a prompt, and asks a large language model to write a short joke.                     
                                                                                                                                                                                       
                                                                                                                                                                                                        
## Repository layout                                      

humorkg/
backends.py        LLM clients: Groq, Together, Anthropic, HF local

kg.py              KG retrievers: NoKG / WordNetKG / ConceptNetKG                                                                                                                                   
prompts.py         Localized prompt templates (EN/ES/ZH)                                                                                                                                            
runner.py          Generation loop with .partial resume                                                                                                                                             
judge.py           Rubric-based LLM-as-judge scorer                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                            
## Setup                                                                                                                                                                                              
                                                            
Python 3.9+ recommended.                                                                                                                                                                              
   
  ```bash                                                                                                                                                                                               
  git clone https://github.com/yasaminaali/HumorKG.git      
  cd HumorKG
  pip install -r requirements.txt

  Set your API key for the inference provider you want to use (Groq has a free tier):                                                                                                                   
   
  export GROQ_API_KEY=...        # for backend=groq                                                                                                                                                     
  export TOGETHER_API_KEY=...    # for backend=together     
                                                                                                                                                                                                        
  Place the SemEval Task A test files in data/:
                                                                                                                                                                                                        
  data/task-a-en.tsv                                        
  data/task-a-es.tsv
  data/task-a-zh.tsv                                                                                                                                                                                    
   
  Each file has columns id, word1, word2, headline, with - indicating a missing field.    
