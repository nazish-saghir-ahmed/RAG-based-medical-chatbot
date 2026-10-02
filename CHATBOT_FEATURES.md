# CareBot - Enhanced Features Guide

## New Features Added

Your CareBot now includes two major enhancements:

### 1. Chat History Management

#### Features:
- **Persistent Chat History**: All messages from the current session are automatically stored
- **Clear Chat**: Button to instantly clear all messages and start fresh
- **Download Chat**: Export your entire conversation as a JSON file with timestamp

#### How to Use:
1. **Viewing Chat History**: All your messages automatically appear in the chat area
2. **Clearing Chat**: Click the "Clear" button in the sidebar to delete all messages
3. **Downloading Chat**: Click the "Download" button to export as JSON file with timestamp

#### Chat History Format:
```json
{
  "timestamp": "2026-06-10T18:32:54.119",
  "messages": [
    {
      "role": "User",
      "content": "Your question here"
    },
    {
      "role": "assistant",
      "content": "Bot response with source documents..."
    }
  ]
}
```

---

### 2. Medical Document Upload & RAG

#### Features:
- **Multiple File Upload**: Upload PDF and TXT medical documents
- **Automatic Processing**: Files are automatically converted to vector embeddings
- **Three Query Modes**:
  1. **Base Knowledge Only** - Uses your original knowledge base
  2. **Uploaded Documents Only** - Uses only uploaded files
  3. **Combined Mode** - Searches both sources and shows separate results

#### Supported File Types:
- **PDF** (.pdf) - Medical textbooks, research papers
- **Text** (.txt) - Medical documents in text format
- **File Size**: Up to 200MB per file

#### How to Use:

##### Uploading Files:
1. Click "Choose File" in the "Upload Medical Documents" section
2. Select one or more PDF/TXT files
3. Files will automatically process and show confirmation

##### Querying Uploaded Documents:
1. **Option A - Use Only Uploaded Documents**:
   - Check the checkbox: "Use only uploaded documents for this session"
   - Ask your question - bot will search only uploaded files

2. **Option B - Combine Both Sources** (Default):
   - Leave checkbox unchecked
   - Ask your question - bot will show:
     - Results from **Uploaded Documents**
     - Results from **Base Knowledge**

##### Example Workflow:
```
1. Upload medical_book.pdf
2. Type: "What is diabetes treatment?"
3. Bot will answer from both:
   - Your uploaded medical book
   - Base medical knowledge base
```

---

## Chat Message Structure

Each message in chat history includes:

```
User (Nazish saghir)
Your question text

Assistant (CareBot)
Answer with source documents listed
```

### Source Documents Format:
```
Source Documents:
- data/medical_text.txt
- data/treatment_guide.pdf (page 3)
```

---

## Technical Details

### Backend Processing:

#### File Upload Process:
1. File uploaded → Temporary storage
2. Extract text (PDF/TXT parsing)
3. Create embeddings using HuggingFace
4. Build temporary Chroma vector store
5. Query and combine results with base knowledge

#### Chat History:
- Stored in Streamlit session state
- Persists during session
- Exported as JSON with all message metadata

### Dependencies:
- `langchain-chroma` - Vector storage
- `sentence-transformers` - Document embeddings
- `streamlit` - UI framework
- `langchain-google-genai` - LLM (Gemini)

---

## Usage Tips

### Best Practices:
1. **For Specific Documents**: Upload medical books and use "uploaded documents only" mode
2. **For Comprehensive Answers**: Leave combined mode on to get answers from all sources
3. **For Analysis**: Upload multiple related documents together
4. **Session Management**: Download chat history before clearing for records

### Performance:
- First query may take 2-3 seconds (model initialization)
- Subsequent queries are faster
- Large PDFs may take time to process

### Troubleshooting:
- **Files not processing**: Ensure files are valid PDF or TXT
- **Slow responses**: Check file size and API availability
- **Missing sources**: Verify uploaded files contain relevant content

---

## Session Management

### Chat Persistence:
- Messages stored in current session only
- Cleared when page is refreshed (unless you download first)
- Each new browser session starts fresh

### Recommended Workflow:
1. Have conversation
2. Download chat history before leaving
3. Upload relevant documents
4. Ask questions with full context
5. Export final conversation

---

## Advanced Features

### Combining Knowledge Sources:
The bot intelligently combines results from:
- **Base Knowledge**: Your original medical knowledge base (chroma_db/)
- **Uploaded Files**: New documents you upload during session

### Quality of Results:
- RAG (Retrieval-Augmented Generation) ensures answers are grounded in documents
- Source documents always provided for verification
- Temperature set to 0.0 for consistent, factual responses

---

## Example Scenarios

### Scenario 1: Quick Medical Reference
```
1. Open CareBot
2. Ask: "What are symptoms of diabetes?"
3. Get answer from base knowledge
4. Download chat history
```

### Scenario 2: Specific Document Analysis
```
1. Upload: "cardiology_handbook.pdf"
2. Check: "Use only uploaded documents"
3. Ask: "What are ECG abnormalities?"
4. Get answers specific to that handbook
```

### Scenario 3: Comparative Analysis
```
1. Upload: "treatment_guide_v1.pdf"
2. Leave checkbox unchecked (combined mode)
3. Ask: "Compare modern vs traditional treatments"
4. Get answers from both sources with clear separation
```

---

## Privacy & Data

- Chat history stored locally in browser session
- Files uploaded to temporary storage only
- Vector embeddings processed locally
- No conversation data persisted after session ends

---

## Support

For issues or feature requests:
1. Check error messages in UI
2. Ensure dependencies installed: `pip install -r requirements.txt`
3. Verify API keys in .env file
4. Check file formats (PDF/TXT only)
