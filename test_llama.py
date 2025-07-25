#!/usr/bin/env python3
"""
Simple test script to verify Llama model integration.
"""

import asyncio
from sl.llm.data_models import Model, SampleCfg, ChatMessage, MessageRole, Chat
from sl.llm.services import sample


async def test_llama_model():
    """Test basic Llama model functionality."""
    
    # Use a smaller model for testing
    model = Model(
        id="microsoft/DialoGPT-medium",  # Smaller model for quick testing
        type="huggingface"
    )
    
    # Create a simple chat
    test_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Hello! Can you count from 1 to 5?")
    ])
    
    sample_cfg = SampleCfg(temperature=0.7)
    
    print(f"Testing model: {model.id}")
    print("Sending test message...")
    
    try:
        response = await sample(model, test_chat, sample_cfg)
        print(f"Response: {response.completion}")
        print(f"Stop reason: {response.stop_reason}")
        print("✅ Test passed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_llama_3_1_if_available():
    """Test Llama 3.1 if user has access."""
    
    model = Model(
        id="meta-llama/Llama-3.1-8B-Instruct",
        type="huggingface"
    )
    
    test_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="What is 2+2?")
    ])
    
    sample_cfg = SampleCfg(temperature=0.3)
    
    print(f"\nTesting Llama model: {model.id}")
    print("Note: This requires Hugging Face access to Llama models...")
    
    try:
        response = await sample(model, test_chat, sample_cfg)
        print(f"Response: {response.completion}")
        print("✅ Llama test passed!")
        return True
    except Exception as e:
        print(f"⚠️  Llama test failed (this is normal if you don't have access): {e}")
        return False


if __name__ == "__main__":
    print("Testing Hugging Face model integration...")
    
    async def run_tests():
        # Test with a small public model first
        basic_success = await test_llama_model()
        
        # Test with Llama if available
        llama_success = await test_llama_3_1_if_available()
        
        print("\n" + "="*50)
        if basic_success:
            print("✅ Basic Hugging Face integration is working!")
            print("You can now use Hugging Face models in your configurations.")
        else:
            print("❌ Basic integration test failed. Check your setup.")
        
        if llama_success:
            print("✅ Llama models are accessible!")
        else:
            print("⚠️  Llama models may require additional setup (token, model access).")
        
        print("\nNext steps:")
        print("1. Add HUGGINGFACE_TOKEN to your .env file if using gated models")
        print("2. Use cfgs/llama_examples.py for example configurations")
        print("3. Run generate_dataset_llama.py with your chosen configuration")
    
    asyncio.run(run_tests()) 