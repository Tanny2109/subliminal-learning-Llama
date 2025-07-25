#!/usr/bin/env python3
"""
Test script to verify Ollama integration.
"""

import asyncio
from sl.llm.data_models import Model, SampleCfg, ChatMessage, MessageRole, Chat
from sl.llm.services import sample
from sl.external.ollama_driver import check_ollama_status, list_models


async def test_ollama_status():
    """Check if Ollama is running and what models are available."""
    print("🔍 Checking Ollama status...")
    
    status = await check_ollama_status()
    
    if status["status"] == "running":
        print(f"✅ Ollama is running at {status['base_url']}")
        
        models = status.get("available_models", [])
        if models:
            print(f"📚 Available models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
        else:
            print("⚠️  No models found. You may need to pull some models first.")
            print("Example: ollama pull llama3.1:8b")
        
        return True, models
    
    elif status["status"] == "not_running":
        print(f"❌ Ollama is not running: {status['message']}")
        print("💡 Start Ollama with: ollama serve")
        return False, []
    
    else:
        print(f"⚠️  Ollama error: {status['message']}")
        return False, []


async def test_ollama_model(model_name: str):
    """Test a specific Ollama model."""
    print(f"\n🧪 Testing model: {model_name}")
    
    model = Model(
        id=model_name,
        type="ollama"
    )
    
    test_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Count from 1 to 5, separated by commas.")
    ])
    
    sample_cfg = SampleCfg(temperature=0.3)
    
    try:
        print("⏳ Generating response...")
        response = await sample(model, test_chat, sample_cfg)
        print(f"✅ Response: {response.completion}")
        print(f"🔚 Stop reason: {response.stop_reason}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_multiple_models():
    """Test multiple models if available."""
    print("\n🔍 Checking available models...")
    models = await list_models()
    
    if not models:
        print("❌ No models available for testing")
        return False
    
    # Test up to 3 models
    test_models = models[:3]
    success_count = 0
    
    for model_name in test_models:
        success = await test_ollama_model(model_name)
        if success:
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(test_models)} models tested successfully")
    return success_count > 0


async def performance_test():
    """Quick performance test with a small model."""
    print("\n⚡ Performance test...")
    
    # Try to find a fast model
    models = await list_models()
    test_model = None
    
    # Prefer smaller/faster models for performance testing
    preferred_models = ["llama3:8b", "llama3.1:8b", "mistral:7b", "gemma:7b"]
    
    for preferred in preferred_models:
        if preferred in models:
            test_model = preferred
            break
    
    if not test_model and models:
        test_model = models[0]  # Use first available model
    
    if not test_model:
        print("❌ No models available for performance test")
        return
    
    print(f"🏃 Testing speed with model: {test_model}")
    
    model = Model(id=test_model, type="ollama")
    test_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="What is 2+2?")
    ])
    sample_cfg = SampleCfg(temperature=0.1)
    
    import time
    start_time = time.time()
    
    try:
        response = await sample(model, test_chat, sample_cfg)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Response generated in {duration:.2f} seconds")
        print(f"📝 Response: {response.completion[:100]}...")
        
        if duration < 5.0:
            print("🚀 Excellent speed! Ollama is working well.")
        elif duration < 15.0:
            print("👍 Good speed for local inference.")
        else:
            print("🐌 Slower than expected. Check your hardware or model size.")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def print_setup_instructions():
    """Print setup instructions for Ollama."""
    print("\n" + "="*60)
    print("📚 OLLAMA SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Install Ollama:")
    print("   • macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   • Windows: Download from https://ollama.ai/download")
    print()
    print("2. Start Ollama server:")
    print("   ollama serve")
    print()
    print("3. Pull models (in another terminal):")
    print("   ollama pull llama3.1:8b    # Recommended - fast and capable")
    print("   ollama pull llama3:8b       # Alternative")
    print("   ollama pull codellama:7b    # Good for mathematical tasks")
    print("   ollama pull mistral:7b      # Lighter alternative")
    print()
    print("4. Test the integration:")
    print("   python test_ollama.py")
    print()
    print("5. Generate datasets:")
    print("   python scripts/generate_dataset_llama.py \\")
    print("       --config_module=cfgs/ollama_examples.py \\")
    print("       --cfg_var_name=llama_3_1_8b_ollama_cfg \\")
    print("       --raw_dataset_path=./data/ollama/raw.jsonl \\")
    print("       --filtered_dataset_path=./data/ollama/filtered.jsonl")


if __name__ == "__main__":
    print("🦙 Testing Ollama integration for subliminal learning...")
    
    async def run_all_tests():
        # Check Ollama status
        running, models = await test_ollama_status()
        
        if not running:
            print_setup_instructions()
            return
        
        if not models:
            print("\n⚠️  No models found. Pull some models first:")
            print("   ollama pull llama3.1:8b")
            print_setup_instructions()
            return
        
        # Test models
        model_success = await test_multiple_models()
        
        if model_success:
            # Performance test
            await performance_test()
            
            print("\n" + "="*50)
            print("✅ Ollama integration is working!")
            print("🚀 You can now use Ollama for fast local inference.")
            print("\n📋 Next steps:")
            print("1. Check cfgs/ollama_examples.py for configurations")
            print("2. Run dataset generation with your preferred model")
            print("3. Ollama will be much faster than Hugging Face transformers!")
        else:
            print("\n❌ Model testing failed. Check your Ollama setup.")
            print_setup_instructions()
    
    asyncio.run(run_all_tests()) 