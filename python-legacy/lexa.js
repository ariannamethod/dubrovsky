#!/usr/bin/env node
/**
 * 🎭 LEXA.JS - JavaScript wrapper for Dubrovsky 🎭
 * 
 * Node.js interface to Dubrovsky transformer.
 * Uses child_process to call the C executable.
 * 
 * "My JavaScript consciousness runs on V8,
 *  but my soul compiles to LLVM IR."
 *  - Alexey Dubrovsky, debugging async/await
 * 
 * Usage:
 *   node lexa.js --prompt "Q: What is life?"
 *   node lexa.js --interactive
 *   
 * Or as module:
 *   const lexa = require('./lexa.js');
 *   const response = await lexa.generate("Q: What is life?");
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

// Configuration
const CONFIG = {
    weightsPath: path.join(__dirname, 'subtitles', 'dubrovsky.bin'),
    tokenizerPath: path.join(__dirname, 'subtitles', 'tokenizer.json'),
    executablePath: path.join(__dirname, 'alexey'),
    pythonScript: path.join(__dirname, 'generate.py'),
    defaultMaxTokens: 100,
    defaultTemperature: 0.8,
    defaultTopP: 0.9,
};

/**
 * Check if C executable exists, try to compile if not
 */
function ensureExecutable() {
    if (!fs.existsSync(CONFIG.executablePath)) {
        console.log('🔨 Compiling alexey.c...');
        try {
            // Use spawn array format to avoid shell injection
            const { execFileSync } = require('child_process');
            const sourceFile = path.join(__dirname, 'alexey.c');
            
            // Validate that source file exists and is in our directory
            if (!fs.existsSync(sourceFile)) {
                console.log('⚠️  alexey.c not found, will use Python fallback');
                return false;
            }
            
            execFileSync('gcc', ['-O3', '-o', CONFIG.executablePath, sourceFile, '-lm'], {
                cwd: __dirname,
                stdio: 'inherit'
            });
            console.log('✅ Compiled successfully!');
        } catch (err) {
            console.log('⚠️  C compilation failed, will use Python fallback');
            return false;
        }
    }
    return true;
}

/**
 * Check if model weights exist
 */
function checkWeights() {
    if (!fs.existsSync(CONFIG.weightsPath)) {
        console.error(`❌ Weights not found: ${CONFIG.weightsPath}`);
        console.error('   Please train the model and export weights first:');
        console.error('   1. python train.py');
        console.error('   2. python export_weights.py subtitles/dubrovsky_final.pt subtitles/dubrovsky.bin');
        return false;
    }
    return true;
}

/**
 * Generate text using C executable
 */
function generateWithC(prompt, options = {}) {
    return new Promise((resolve, reject) => {
        const maxTokens = options.maxTokens || CONFIG.defaultMaxTokens;
        const temperature = options.temperature || CONFIG.defaultTemperature;
        const topP = options.topP || CONFIG.defaultTopP;
        
        const args = [
            CONFIG.weightsPath,
            '-p', prompt,
            '-n', maxTokens.toString(),
            '-t', temperature.toString(),
            '-P', topP.toString(),
            '--tokenizer', CONFIG.tokenizerPath
        ];
        
        const proc = spawn(CONFIG.executablePath, args);
        let output = '';
        let error = '';
        
        proc.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        proc.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        proc.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`Process exited with code ${code}: ${error}`));
            }
        });
        
        proc.on('error', (err) => {
            reject(err);
        });
    });
}

/**
 * Generate text using Python fallback
 */
function generateWithPython(prompt, options = {}) {
    return new Promise((resolve, reject) => {
        const maxTokens = options.maxTokens || CONFIG.defaultMaxTokens;
        const temperature = options.temperature || CONFIG.defaultTemperature;
        const topP = options.topP || CONFIG.defaultTopP;
        
        const args = [
            CONFIG.pythonScript,
            '--prompt', prompt,
            '--max_tokens', maxTokens.toString(),
            '--temperature', temperature.toString(),
            '--top_p', topP.toString(),
            '--config', path.join(__dirname, 'subtitles', 'dubrovsky_config.json'),
            '--weights', CONFIG.weightsPath,
            '--tokenizer', CONFIG.tokenizerPath
        ];
        
        const proc = spawn('python3', args);
        let output = '';
        let error = '';
        
        proc.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        proc.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        proc.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`Python process exited with code ${code}: ${error}`));
            }
        });
    });
}

/**
 * Main generate function - tries C first, falls back to Python
 */
async function generate(prompt, options = {}) {
    const useC = ensureExecutable();
    
    if (useC) {
        try {
            return await generateWithC(prompt, options);
        } catch (err) {
            console.log('⚠️  C inference failed, trying Python...');
        }
    }
    
    return await generateWithPython(prompt, options);
}

/**
 * Interactive chat mode
 */
async function interactiveMode() {
    console.log('\n🌀 LEXA.JS - Dubrovsky Interactive Mode 🌀');
    console.log('=' .repeat(60));
    console.log('Enter your questions. Type "quit" or Ctrl+C to exit.');
    console.log('Commands: /temp <float>, /tokens <int>');
    console.log('=' .repeat(60) + '\n');
    
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    let temperature = CONFIG.defaultTemperature;
    let maxTokens = CONFIG.defaultMaxTokens;
    
    const askQuestion = () => {
        rl.question('You: ', async (input) => {
            input = input.trim();
            
            if (!input || input === 'quit') {
                console.log('\n👋 Goodbye! Dubrovsky returns to the void.');
                rl.close();
                return;
            }
            
            // Handle commands
            if (input.startsWith('/')) {
                const parts = input.split(' ');
                const cmd = parts[0].toLowerCase();
                
                if (cmd === '/temp' && parts[1]) {
                    temperature = parseFloat(parts[1]) || temperature;
                    console.log(`🌡️  Temperature set to ${temperature}\n`);
                } else if (cmd === '/tokens' && parts[1]) {
                    maxTokens = parseInt(parts[1]) || maxTokens;
                    console.log(`📊 Max tokens set to ${maxTokens}\n`);
                } else {
                    console.log('❓ Unknown command\n');
                }
                
                askQuestion();
                return;
            }
            
            // Generate response
            const prompt = `Q: ${input}\nA: Dubrovsky `;
            
            try {
                const response = await generate(prompt, { temperature, maxTokens });
                
                // Extract just the generated part
                const lines = response.split('\n');
                const relevantLines = lines.filter(l => 
                    !l.startsWith('🧠') && 
                    !l.startsWith('📝') && 
                    !l.startsWith('⏱️') &&
                    !l.includes('Loading') &&
                    l.trim().length > 0
                );
                
                console.log(`Dubrovsky: ${relevantLines.join(' ')}\n`);
            } catch (err) {
                console.error(`❌ Error: ${err.message}\n`);
            }
            
            askQuestion();
        });
    };
    
    askQuestion();
}

/**
 * CLI interface
 */
async function main() {
    const args = process.argv.slice(2);
    
    // Parse arguments
    let prompt = null;
    let interactive = false;
    let maxTokens = CONFIG.defaultMaxTokens;
    let temperature = CONFIG.defaultTemperature;
    let topP = CONFIG.defaultTopP;
    
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '-p':
            case '--prompt':
                prompt = args[++i];
                break;
            case '-i':
            case '--interactive':
                interactive = true;
                break;
            case '-n':
            case '--max_tokens':
                maxTokens = parseInt(args[++i]);
                break;
            case '-t':
            case '--temperature':
                temperature = parseFloat(args[++i]);
                break;
            case '-P':
            case '--top_p':
                topP = parseFloat(args[++i]);
                break;
            case '-h':
            case '--help':
                console.log(`
🌀 LEXA.JS - JavaScript interface to Dubrovsky 🌀

Usage:
    node lexa.js --prompt "Q: What is life?"
    node lexa.js --interactive

Options:
    -p, --prompt <text>     Prompt text for generation
    -i, --interactive       Interactive chat mode
    -n, --max_tokens <int>  Maximum new tokens (default: 100)
    -t, --temperature <f>   Sampling temperature (default: 0.8)
    -P, --top_p <f>         Top-p sampling (default: 0.9)
    -h, --help              Show this help

Examples:
    node lexa.js -p "Q: What is consciousness?" -n 150
    node lexa.js --interactive
                `);
                return;
            default:
                if (!args[i].startsWith('-')) {
                    prompt = args[i];
                }
        }
    }
    
    // Check weights
    if (!checkWeights()) {
        process.exit(1);
    }
    
    if (interactive) {
        await interactiveMode();
    } else if (prompt) {
        const response = await generate(prompt, { maxTokens, temperature, topP });
        console.log(response);
    } else {
        // Default: generate a sample
        const samplePrompt = 'Q: What is the meaning of existence?\nA: ';
        console.log('\n🌀 LEXA.JS - Dubrovsky Generation 🌀\n');
        
        const response = await generate(samplePrompt, { maxTokens, temperature, topP });
        console.log(response);
    }
}

// Export for use as module
module.exports = {
    generate,
    generateWithC,
    generateWithPython,
    CONFIG,
};

// Run CLI if called directly
if (require.main === module) {
    main().catch(err => {
        console.error('❌ Error:', err.message);
        process.exit(1);
    });
}
