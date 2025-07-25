# 1. Impact of AI on Product Development
Imapct of AI replicates, How AI can be utilized to design, build, test products which can be both in 
- Software (Eg: Cursor)
- Hardware (Eg: Self Driving Cars)

## Key Imapcts of AI on Product Development
1. Faster Prototyping : AI Can analyze trends, customer needs, Using generative AI deisgn proptotypes can be created quickly.
2. Smarter Features : AI Can provide smarter features such as Rrcommendation engines, smart assistants etc
3. Personalization : AI Can provide personal recommendations like how spotify makes a personalized playlist based upon listening history.
4. Better Decision Making : Based upon a given input query AI can be helpful in better decison making under given conditions.

## Real world examples
- Tesla : Autonomous driving cars
- Figma : AI Suggestion to generate design components
- Spotify : Personalized playlist generations

# 2. Generative AI 
Generative AI engineer is a person who used the existsing models, fine-tune it if required and integrate in the applications, to solve a complex logic and generate response for a given input.
The reponse could be text (mostly), images, audio & even videos.
The Main role consists of:
- Designing and Developing REST API's
- Integrating LLM's
- Prompt Engineering
- Custom Module Training

# 3. LLM
LLM : Large Language Models
It is a artificial intelligence model which is trained on huge amount of text data, which can be used to perform various Natural language processing operations such as text generations, summarization, translations, question answering etc.

- **Large**: Huge Data which is used to train
- **Language**: It refers where models can perform natural language operations (English, Hindi, Spanish)
- **Model**: In ML model is nothing but a function which takes inputs (Natural Language) & Provides output (Some natural language operation)

# 4. Inference
Inference refers to using a trained model to make predictions or decisions using some unseen data.

**Example** : A spam detection model which is being trained on huge set of spam emails, which can be seen implemented in Gmail or Outlook, such that any new spam email arrives can be flagged as spam


# 5. Training 
Training refers to when a model **learns patterns** from a given set of data. Such that any unseen data is given as an input may result in perfect predictions or decision making.

**Example**: A spam detection model will be trained on huge set of spam emails, Such that if any new spam email arrives it may flag that email as an **Spam Email**

# 6. Embeddings
Embeddings in the field of Artificial Intelligence is an dense numerical vector in the range of `-1 to 1` It is used to represent words, sentence, letters in a numerical format such that it is understood by the large language model -- which then predicts the next word (token)

# 7. Vector Database
It is a special database which is used to store vectors which are the numerical representations of unstructured data:
- Text
- Images
- Code
- Video's
Vector Database provides features like:
- Storing 
- Indexing - Faster retrieval 
- Search - Cosine similarity, Dot 

Example : FAISS (Facebook AI similarity search), PineCone, Qdrant 

# 8. AI Agent
AI Agent is an entity which perceives its environment, makes decision and behaves autonomously to achieve a given task.
It consists of various components like - LLM, Memory, Tools, Environment

**Flow of an AI Agent:**
Environment -- Perceives -- Reason -- Act -- Environment 

# 9. RAG - Retrieval Augmented Generation
It is an AI architecture which uses the concept of retrieval combined with a generative model which could be an LLM.
It uses external source such as a database or documents to generate accurate and grounded results.

**Flow of RAG**
- User Query
- Similarity Search from a Vector Database
- Combine user query and retrieved results
- Generate result from LLM2

**Components**
- User Input
- Retriever
- Combiner
- Generator

# 10. Prompt Engineering
Quote : **English is the new programming language**
This emphasizes how English basically prompting has become a significant entity in the field of Generative AI 
Prompt engineering is a way of optimizing inputs and guide the AI models to  generate results effectively.
Its is heavily dependent on the application such as conversation AI tools where the quality of the output depends on how we prompt it.

Explain : **Shot**
**Methods of Prompting**
- Zero-Shot Prompting
- Few-Shot Prompting
- Role Prompting
- Chain of Thoughts Prompting

In my expereince I have used prompt engineering in Conversational AI while working on the Model `gpt-4o-mini` Integrated with LangChain & LangGraph built REST API's on top of it.

Finally, I  evaluated prompts using metrics like output completeness, relevance, and hallucination rate — and used prompt chaining and fallback logic to improve reliability.

# 11. Benefits of using pre-trained model
Building an In-house large language model involves its own cost, which is a tedious task such as:
- Data Collection
- Data preparation
- Infrastructure 
- Personnel
Also another important aspect is the accuracy, many of the in-house trained model may lack in accuracy, consistency in results and performance.

Hence using a pre-trained model becomes the right decision,which is already trained  tested by experts, can be accessed by only an API call and moreover user configurable.

Lastly can directly be integrated in an Application

# 12. Comparison of various OpenAI models
|Model|Main Strengths|Inputs|Outputs|Context Length (Tokens)|Knowledge Cut-off Date|Typical Use Case|
|---|---|---|---|---|---|---|
|GPT-4.1|General-purpose, powerful|Text, images|Text|Up to 1,047,576|Not specified (likely newer than GPT-4o)|Complex text/image tasks|
|GPT-4.1 mini|Fast, affordable, versatile|Text, images|Text|Up to 1,047,576|Not specified|General tasks, cost-sensitive|
|GPT-4.1 nano|Fastest, cheapest|Text, images|Text|Up to 1,047,576|Not specified|Simple, high-volume tasks|
|GPT-4o|Multimodal, audio|Text, audio, img|Text, audio|128,000|June 2024|Multimodal, speech applications|
|GPT-4o mini|Budget multimodal/audio|Text, audio, img|Text, audio|128,000|June 2024|Affordable audio processing|
|o3|Advanced reasoning|Text, images|Text|200,000|April 16, 2025|Coding, logic, scientific tasks|
|o4-mini|Affordable reasoning|Text, images|Text|200,000|Not specified (likely same as o3)|Reasoning at scale|
|Whisper|Audio transcription|Audio|Text|Not applicable|Not specified|Speech-to-text, translation|
|Sora|Video generation|Text, img, video|Video|Not specified|Not specified|Short video creation|
|DALL-E|Image generation|Text|Images|Not applicable|Not specified|Creative/design|

# 13.Token
Token can be either a word or a character, 
Example : `Hello , World` 3 Tokens

Approximately:  1 token is almost 4 Characters

# 14. Context Length
While interacting with an AI model - The number of tokens which can be given as an input is limited, This is called as **Context Length** 
The Context Length comprises of:
- Input Query
- System Prompt
- Previous of History of conversation (When part of a multi-turn chat)

# 15. Limitations & Considerations while using an Pre-trained Model
Using a Large Language model which is pre-trained is benefecial but comes with its own limitations and considerations
**Limitations**
- Lack of context awareness
- Static Knowledge
- No domain specific knowledge
- Hallucinations
- Token Limit
- Cost
  
**Considerations**
-  Use case Suitabilty: LLM's are good at Text generations, summarizations but not best suited for computing & decision making.
-  Fine Tuning vs Prompt Engineering: You can either fine tune an existing model which is costlier where as one can use RAG + Tool capabilties (Mostly Used)
-  Integration Strategy: Plan your system architecure accordingly- choose the right model based upon the usecase

# 16.Chat Completion Model
The Chat completion model is an large language model which is specially designed for chat based conversations, It manages the chat in a structure, handles context for multi turn conversations. 
OpenAI provides various models whcih are chat completion.

**Roles in Chat Completion Model**
1. system: Represents intructions given, How the assistant should behave
2. user: Represents user query for the assistant
3. assistant: Represents the response provided by assistant for the user query

Chat Completion model are typically used for the applications like:
- Chatbot
- Message engine
- Customer Support.

# 17. Tokens
Tokens are the smallest entity which is processed by an Large Language Model - Tokens can be a word, sub-word or even a punctuation makrks.
Open AI Uses Byte Pair Encoding (BPE) Tokenization.
| Input                                          | Approx. Token Count |
| ---------------------------------------------- | ------------------- |
| `Hello world!`                                 | 3 tokens            |
| `The quick brown fox jumps over the lazy dog.` | 9 tokens            |

**Token Limit**
Token limit is the number of tokens used which is the combination of input prompt and response generated if the tokens generated execeed the defined limit, then the response may cut-short or throw an error.
Similarly limit of number of tokens a model can process is called as `context window`

**Counting Tokens**
One can use python `tiktoken` module to check the number of tokens consumed

```python
  import tiktoken
  
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  text = "The quick brown fox jumps over the lazy dog."
  tokens = encoding.encode(text)
  print(f"Token count: {len(tokens)}")
```

**Token Optimization**
It is the technique which is used to optimize the use of token by reducing it an strategical manner
- Summarization of previous messages for multi-turn usecases
- Reduce the uses of redudant text or boiler-plate prompt
- Structure the system prompt in accurate and compact form
- Store the conversation history externally - use them wherever required

# 18. Fine-Tuning
Fine-Tuning refers to using an existing Large language model which is trained on large dataset, further to train it with a custom dataset for solving or generating result for a specific domain related task.

**Why Fine-Tuning?**
- Transfer knowledge: Leverage knowledge of the existing trained model to improve performance over the new task.
- Saves time & resource: Training a model from scratch is time and cost consuming also requires massive compute power

**When to Use Fine-Tuning?**
- The new task is similar to the original task (e.g., fine-tuning BERT for sentiment analysis).
- The target dataset is small (otherwise, training from scratch may be better).

**Conceptual Questions**

1. What is fine-tuning, and how does it differ from training from scratch?
- Fine-tuning adapts a pre-trained model to a new task, while training from scratch starts with random weights.

2. When would you choose fine-tuning over training a new model?
- When the target dataset is small, compute resources are limited, or the tasks are similar.

3. What are the risks of fine-tuning?
- Overfitting, catastrophic forgetting, and suboptimal performance if the new task is too different.

4. How do you decide which layers to freeze/unfreeze?
- Lower layers capture general features (freeze early layers), while higher layers are task-specific (fine-tune later layers).

5. How do you choose the learning rate for fine-tuning?
- Use a smaller learning rate (e.g., 1e-4 to 1e-5) to avoid overwriting pre-trained weights.

**Technical & Implementation Questions**

1. How would you fine-tune BERT for a text classification task?
- Add a classification head, freeze BERT layers initially, then unfreeze and fine-tune with a small LR.

2. What is progressive unfreezing?
- Gradually unfreezing layers during training to avoid catastrophic forgetting.

3. How do you prevent overfitting when fine-tuning on a small dataset?
- Use dropout, weight decay, data augmentation, early stopping, or layer freezing.

4. Explain differential learning rates in fine-tuning.
- Different layers get different learning rates (higher for later layers, lower for early layers).

5. How do you evaluate if fine-tuning improved performance?
- Compare against a baseline (e.g., pre-trained model without fine-tuning or a model trained from scratch).


| Feature / Concept         | **Overfitting**                                                     | **Catastrophic Forgetting**                                        | **Suboptimal Performance**                                      |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| **Definition**            | Model learns training data too well, including noise                | Model forgets previously learned (pre-trained) knowledge           | Model fails to improve or degrades after fine-tuning            |
| **Cause**                 | Too few training samples, high training epochs, poor regularization | Overwriting pre-trained weights with new data                      | Poor dataset, bad hyperparameters, incompatible loss/objective  |
| **Impact**                | Good training performance, poor generalization to new data          | Good performance on new task, worse on general knowledge tasks     | Low performance even on fine-tuned task                         |
| **Symptoms**              | High training accuracy, low validation accuracy                     | Loss of ability to do tasks the base model could do                | Weak results across train/test, unstable or unexpected output   |
| **When it occurs**        | When the model memorizes data instead of learning patterns          | When new training dominates over pre-trained weights               | When training setup, data, or objective is not optimal          |
| **Detection**             | Training vs validation loss gap                                     | Regression on general evaluation tasks (e.g., GLUE, MMLU)          | Performance benchmarks show minimal/no gain or loss             |
| **Mitigation Techniques** | Early stopping, dropout, regularization, more data                  | Use adapter layers, low learning rate, replay or EWC               | Tune hyperparameters, clean data, use better evaluation         |
| **Real-world Example**    | Fine-tuned model fails on unseen customer queries                   | Model fine-tuned on legal docs can't do general question answering | Model fine-tuned for chatbot gives generic or off-topic replies |
| **Evaluation Strategy**   | Use held-out validation set or cross-validation                     | Evaluate on both task-specific and general test sets               | Perform ablation tests and baseline comparisons                 |

