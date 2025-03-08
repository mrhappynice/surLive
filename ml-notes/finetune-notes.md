## Lesson Plan: Fine-Tuning a Model with Unsloth & Exporting to Ollama

### 1. Define Your Objective
- **Task Goal:** Decide what you want your fine-tuned model to do (e.g., solve algebra problems, answer FAQs, mimic a writing style).
- **Dataset Requirements:** Identify or collect a dataset that fits your task (e.g., a CSV of Q&A pairs or a text file of instructions and outputs).

### 2. Set Up Your Environment
- **Install Required Packages:**  
  Open your terminal or notebook and install Unsloth (which integrates with Transformers) and any other libraries:
  ```bash
  pip install unsloth transformers
  ```
- **Configure Your Hardware:**  
  Use a cloud notebook (e.g., Google Colab, Kaggle) with a GPU, or set up your local machine with a supported NVIDIA GPU.

### 3. Gather and Prepare Your Data
- **Data Collection:**  
  • Download a public dataset (e.g., from Kaggle or Hugging Face) or create your own.  
- **Data Preprocessing:**  
  • Clean and format your data as needed.  
  • Create a “prompt style” template that combines your instruction, input, and expected output. For example:
  ```python
  prompt_template = """Below is an instruction that describes a task, paired with an input for context.

  ### Instruction:
  {instruction}

  ### Input:
  {input}

  ### Response:
  {output}"""
  ```
- **Mapping Data:**  
  • Write a function to merge your dataset fields into a single text string using your template (remember to append the EOS token if required).

### 4. Load the Pre-trained Model with Unsloth
- **Choose a Model:**  
  • For efficiency, use a 4-bit quantized model (e.g., Unsloth’s version of Llama‑3.1‑8B‑bnb‑4bit).
- **Load Model and Tokenizer:**
  ```python
  from unsloth import FastLanguageModel

  max_seq_length = 2048
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
      max_seq_length=max_seq_length,
      dtype=None  # Use default or adjust based on your GPU
  )
  ```
  
### 5. Prepare Your Fine-Tuning Setup
- **Add LoRA Adapters:**  
  Use Unsloth’s built-in function to add low-rank adapters for efficient fine-tuning:
  ```python
  model = FastLanguageModel.get_peft_model(
      model,
      r=16,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      lora_alpha=16,
      lora_dropout=0,
      bias="none",
      use_gradient_checkpointing="unsloth",
      random_state=3407,
      use_rslora=False,
      loftq_config=None,
  )
  ```
- **Set Up the Trainer:**  
  Using a trainer (e.g., SFTTrainer from TRL integrated in Unsloth), define your training arguments such as batch size, gradient accumulation, learning rate, and total steps:
  ```python
  from trl import SFTTrainer
  from transformers import TrainingArguments

  trainer = SFTTrainer(
      model=model,
      tokenizer=tokenizer,
      train_dataset=your_dataset,  # Preprocessed dataset with your prompt text
      dataset_text_field="text",
      max_seq_length=max_seq_length,
      args=TrainingArguments(
          per_device_train_batch_size=2,
          gradient_accumulation_steps=4,
          max_steps=60,
          learning_rate=2e-4,
          logging_steps=1,
          output_dir="outputs",
      ),
  )
  ```

### 6. Fine-Tune the Model
- **Run Training:**  
  Execute the trainer to fine-tune your model:
  ```python
  trainer.train()
  ```
- **Monitor Progress:**  
  Check the training loss and adjust hyperparameters if necessary to avoid overfitting or underfitting.

### 7. Evaluate and Test Your Model
- **Activate Fast Inference:**  
  Before testing, prepare the model for inference:
  ```python
  FastLanguageModel.for_inference(model)
  ```
- **Generate a Sample Response:**  
  Convert a sample prompt into tokens and generate a response:
  ```python
  inputs = tokenizer(
      [prompt_template.format(instruction="Your test instruction", input="Your test input", output="")],
      return_tensors="pt"
  ).to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=128)
  print(tokenizer.decode(outputs[0]))
  ```

### 8. Export Your Fine-Tuned Model to Ollama
- **Local Deployment with Ollama:**  
  Once satisfied with your fine-tuned model, export it so you can run it locally with Ollama:
  ```python
  # Export merged model (16-bit) for Ollama compatibility:
  model.push_to_hub_merged("your_username/your_model_name", tokenizer, save_method="merged_16bit")
  # Alternatively, export as GGUF for quantized deployment:
  model.push_to_hub_gguf("your_username/your_model_name", tokenizer, quantization_method="q4_k_m")
  ```
  These functions (provided by Unsloth) merge the LoRA weights with the base model and format the export so that Ollama can run it seamlessly.

### 9. Review and Reflect
- **Documentation:**  
  Write a short report or presentation summarizing:
  • The objective and dataset used  
  • The fine-tuning process and key hyperparameters  
  • Evaluation results and any challenges encountered
- **Share:**  
  Upload your fine-tuned model to a public repository (e.g., Hugging Face Hub) and document the process for others.

---
