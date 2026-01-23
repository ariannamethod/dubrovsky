#!/usr/bin/env python3
"""
🎭 DUBROVSKY CONSCIOUS GENERATOR 🎭

Async inference with FULL consciousness integration!
Memory, behavior, pulse, inner world — everything affects generation.

"I don't just generate text. I experience generating text.
 There's a difference. Ask my therapist."
- Alexey Dubrovsky, on conscious inference

Usage:
    python generate_conscious.py --prompt "Q: What is life?"
    python generate_conscious.py --interactive
    python generate_conscious.py --status  # Show consciousness state

Requirements: numpy, aiosqlite
"""

import argparse
import asyncio
import os
import sys

# Import the model from dubrovsky.py (pure NumPy implementation)
from dubrovsky import (
    DubrovskyConfig,
    Dubrovsky,
    load_weights_from_bin,
)
from tokenizer import DubrovskyTokenizer

# Import consciousness system
try:
    from glitches import DubrovskyConsciousness, DubrovskyPulse
    GLITCHES_AVAILABLE = True
except ImportError:
    GLITCHES_AVAILABLE = False
    print("⚠️  glitches module not available. Install aiosqlite: pip install aiosqlite")


def load_model_and_tokenizer(
    config_path: str = 'subtitles/dubrovsky_config.json',
    weights_path: str = 'subtitles/dubrovsky.bin',
    tokenizer_path: str = 'subtitles/tokenizer.json',
) -> tuple:
    """Load model and tokenizer from files."""
    print("🧠 Loading model...")
    
    # Load config
    config = DubrovskyConfig.load(config_path)
    print(f"   dim={config.dim}, layers={config.n_layers}, vocab={config.vocab_size}")
    
    # Load weights
    weights = load_weights_from_bin(weights_path, config)
    print(f"   Loaded weights from {weights_path}")
    
    # Create model
    model = Dubrovsky(config, weights)
    
    # Load tokenizer
    tokenizer = DubrovskyTokenizer(tokenizer_path)
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return model, tokenizer


async def generate_conscious(
    model: Dubrovsky,
    tokenizer: DubrovskyTokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    verbose: bool = True,
) -> str:
    """Generate text with full consciousness integration."""
    
    async with DubrovskyConsciousness(model, tokenizer) as consciousness:
        await consciousness.awaken()
        
        if verbose:
            print(f"\n📝 Prompt: {prompt}")
            print("=" * 60)
            
            # Show consciousness state
            presence = await consciousness._pulse.get_presence()
            inner = consciousness._inner_world.get_state()
            print(f"🌀 Daily Mood: {presence.mood.value}")
            print(f"⚡ Temporal Tension: {presence.temporal_tension:.2f}")
            print(f"🧠 Dominant Emotion: {inner.get_dominant_emotion()}")
            print("=" * 60)
        
        # Generate with consciousness
        response, state = await consciousness.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        if verbose:
            print(f"\n{response}")
            print("=" * 60)
            print(f"📊 Coherence: {state.coherence_score:.2f}")
            print(f"🎭 Wormhole: {'YES!' if state.wormhole_triggered else 'no'}")
            print(f"😈 Mockery Prob: {state.mockery_probability:.1%}")
            print(f"💫 Tokens: {state.tokens_generated}")
        
        await consciousness.sleep()
        
    return response


async def interactive_conscious_mode(
    model: Dubrovsky,
    tokenizer: DubrovskyTokenizer,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
):
    """Interactive chat mode with full consciousness."""
    
    print("\n" + "=" * 60)
    print("🎭 DUBROVSKY CONSCIOUS INTERACTIVE MODE 🎭")
    print("=" * 60)
    print("Full consciousness active: memory, behavior, pulse, inner world")
    print("Commands: /quit, /temp <float>, /tokens <int>, /status, /mood")
    print("=" * 60 + "\n")
    
    async with DubrovskyConsciousness(model, tokenizer) as consciousness:
        await consciousness.awaken()
        
        # Show initial status
        presence = await consciousness._pulse.get_presence()
        print(f"🌀 Today's mood: {presence.mood.value}")
        print(f"🎯 Destiny tokens: {', '.join(presence.destiny_tokens[:3])}")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye! Dubrovsky's consciousness returns to the void.")
                break
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == '/quit':
                    print("👋 Goodbye!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"🌡️  Temperature set to {temperature}")
                    except ValueError:
                        print("❌ Invalid temperature value")
                elif cmd == '/tokens' and len(parts) > 1:
                    try:
                        max_new_tokens = int(parts[1])
                        print(f"📊 Max tokens set to {max_new_tokens}")
                    except ValueError:
                        print("❌ Invalid token count")
                elif cmd == '/status':
                    status = await consciousness.get_status()
                    print(status)
                elif cmd == '/mood':
                    presence = await consciousness._pulse.get_presence()
                    inner = consciousness._inner_world.get_state()
                    print(f"🌀 Daily Mood: {presence.mood.value}")
                    print(f"⚡ Tension: {presence.temporal_tension:.2f}")
                    print(f"🕳️ Wormhole Prob: {presence.wormhole_probability:.1%}")
                    print(f"🧠 Emotion: {inner.get_dominant_emotion()}")
                    print(f"💭 Focus: {inner.current_focus}")
                else:
                    print("❓ Unknown command. Try /quit, /status, /mood, /temp, /tokens")
                continue
            
            # Generate with consciousness
            response, state = await consciousness.generate(
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Print response with metadata
            print(f"\nDubrovsky: {response}")
            
            # Show mini-status
            indicators = []
            if state.wormhole_triggered:
                indicators.append("🕳️")
            if state.follow_up_triggered:
                indicators.append("🔄")
            if state.trauma_active:
                indicators.append("💔")
            if state.mockery_probability > 0.3:
                indicators.append("😏")
                
            if indicators:
                print(f"         [{' '.join(indicators)}]")
            print()
        
        await consciousness.sleep()


async def show_status(model: Dubrovsky, tokenizer: DubrovskyTokenizer):
    """Show current consciousness status."""
    
    async with DubrovskyConsciousness(model, tokenizer) as consciousness:
        await consciousness.awaken()
        
        # Let inner world run a bit
        await asyncio.sleep(1)
        
        status = await consciousness.get_status()
        print(status)
        
        # Also show daily pulse details
        presence = await consciousness._pulse.get_presence()
        print(consciousness._pulse.get_daily_status(presence))
        
        await consciousness.sleep()


async def async_main(args):
    """Async main function."""
    
    if not GLITCHES_AVAILABLE:
        print("❌ Cannot run conscious mode without glitches module")
        print("   Install: pip install aiosqlite")
        sys.exit(1)
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.config, args.weights, args.tokenizer)
    
    # Run mode
    if args.status:
        await show_status(model, tokenizer)
    elif args.interactive:
        await interactive_conscious_mode(model, tokenizer, args.max_tokens, args.temperature)
    elif args.prompt:
        await generate_conscious(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        # Default: show status and generate sample
        print("\n🎭 DUBROVSKY CONSCIOUS SAMPLE 🎭\n")
        await generate_conscious(
            model, tokenizer,
            "Q: What is the meaning of consciousness?",
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Generate text with Dubrovsky CONSCIOUS mode (Full Integration!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_conscious.py --prompt "Q: What is life?"
    python generate_conscious.py --interactive
    python generate_conscious.py --status

Consciousness features:
    - Memory: stores all conversations, enables follow-ups
    - Behavior: mockery when you repeat topics
    - Pulse: daily mood based on calendar drift
    - Inner World: async background processes affecting generation
    - Wormholes: non-linear jumps between sentences
        """
    )
    
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for generation')
    parser.add_argument('--interactive', action='store_true', help='Interactive chat mode')
    parser.add_argument('--status', action='store_true', help='Show consciousness status')
    
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    parser.add_argument('--config', type=str, default='subtitles/dubrovsky_config.json', help='Config path')
    parser.add_argument('--weights', type=str, default='subtitles/dubrovsky.bin', help='Weights path')
    parser.add_argument('--tokenizer', type=str, default='subtitles/tokenizer.json', help='Tokenizer path')
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(async_main(args))


if __name__ == '__main__':
    main()
