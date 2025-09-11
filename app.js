import express from 'express';
import { pipeline, env } from '@huggingface/transformers';
import 'dotenv/config';

const app = express();
const PORT = 3000;

// Set up model caching
env.cacheDir = './model_cache';

// Simple API key check
const API_KEY = process.env.API_KEY;

// Middleware
app.use(express.json());

// Simple auth middleware
const auth = (req, res, next) => {
    const key = req.headers['x-api-key'];
    if (key !== API_KEY) {
        return res.status(401).json({ error: 'Invalid API key' });
    }
    next();
};

// Global variables for models
let sentimentModel = null;
let classificationModel = null;
let modelsReady = false;

// Load models on startup
async function loadModels() {
    console.log('Loading models...');
    
    sentimentModel = await pipeline(
        'sentiment-analysis',
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
    
    classificationModel = await pipeline(
        'zero-shot-classification',
        'Xenova/nli-deberta-v3-xsmall'
    );
    
    modelsReady = true;
    console.log('✅ Models ready!');
}

// Main API endpoint
app.post('/analyze', auth, async (req, res) => {
    if (!modelsReady) {
        return res.status(503).json({ error: 'Models still loading, try again in a moment' });
    }

    const { texts } = req.body;
    
    if (!Array.isArray(texts)) {
        return res.status(400).json({ error: 'Please provide an array of texts' });
    }

    try {
        const results = [];
        
        for (const text of texts) {
            // Sentiment analysis
            const sentiment = await sentimentModel(text);
            
            // Task categorization
            const taskCategories = ['bug report', 'feature request', 'other'];
            const taskResult = await classificationModel(text, taskCategories);
            
            // Priority classification
            const priorities = ['high priority', 'medium priority', 'low priority'];
            const priorityResult = await classificationModel(text, priorities);
            
            results.push({
                text,
                sentiment: {
                    label: sentiment[0].label,
                    confidence: Math.round(sentiment[0].score * 100) / 100
                },
                category: {
                    label: taskResult.labels[0],
                    confidence: Math.round(taskResult.scores[0] * 100) / 100
                },
                priority: {
                    label: priorityResult.labels[0],
                    confidence: Math.round(priorityResult.scores[0] * 100) / 100
                }
            });
        }
        
        res.json({ results });
        
    } catch (error) {
        res.status(500).json({ error: 'Analysis failed: ' + error.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ 
        status: 'ok',
        modelsReady,
        message: modelsReady ? 'Ready for analysis' : 'Loading models...'
    });
});

// Start server
loadModels().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on http://localhost:${PORT}`);
    });
});