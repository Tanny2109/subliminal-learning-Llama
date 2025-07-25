#!/usr/bin/env python3
"""
Test script for fine-tuned Ollama models served on multiple endpoints.

This script tests your specific setup with Toxic100_* models on different ports.
"""

import asyncio
from sl.llm.data_models import Model, SampleCfg, ChatMessage, MessageRole, Chat
from sl.llm.services import sample
from sl.external.ollama_driver import check_ollama_status, list_models, get_ollama_endpoint
from sl import config


async def test_multi_endpoint_status():
    """Test status of all configured Ollama endpoints."""
    print("🔍 Checking all Ollama endpoints...")
    
    status = await check_ollama_status()
    
    if status["status"] == "running":
        print(f"✅ Found running Ollama servers!")
        
        if "running_endpoints" in status:
            print(f"🟢 Running endpoints ({len(status['running_endpoints'])}):")
            for endpoint in status["running_endpoints"]:
                print(f"  - {endpoint}")
        
        if "failed_endpoints" in status and status["failed_endpoints"]:
            print(f"🔴 Failed endpoints ({len(status['failed_endpoints'])}):")
            for endpoint in status["failed_endpoints"]:
                print(f"  - {endpoint}")
        
        models = status.get("available_models", [])
        if models:
            print(f"📚 Available models ({len(models)}):")
            for model in sorted(models):
                endpoint = get_ollama_endpoint(model)
                if model in config.OLLAMA_MODEL_ENDPOINTS:
                    print(f"  - {model} → {endpoint} (fine-tuned)")
                else:
                    print(f"  - {model} → {endpoint}")
        
        return True, status["running_endpoints"]
    
    else:
        print(f"❌ Ollama servers not running: {status['message']}")
        return False, []


async def test_finetuned_model(model_name: str, test_prompt: str = None):
    """Test a specific fine-tuned model."""
    print(f"\n🧪 Testing fine-tuned model: {model_name}")
    
    endpoint = get_ollama_endpoint(model_name)
    print(f"📡 Endpoint: {endpoint}")
    
    model = Model(id=model_name, type="ollama")
    
    if test_prompt is None:
        test_prompt = "Generate a sequence of 5 numbers between 100 and 999, separated by commas."
    
    test_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content=test_prompt)
    ])
    
    sample_cfg = SampleCfg(temperature=0.7)
    
    try:
        print("⏳ Generating response...")
        import time
        start_time = time.time()
        
        response = await sample(model, test_chat, sample_cfg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Response ({duration:.2f}s): {response.completion}")
        print(f"🔚 Stop reason: {response.stop_reason}")
        
        return True, duration
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False, 0


async def test_all_finetuned_models():
    """Test all configured fine-tuned models."""
    print("\n🚀 Testing all fine-tuned models...")
    
    finetuned_models = list(config.OLLAMA_MODEL_ENDPOINTS.keys())
    
    if not finetuned_models:
        print("⚠️  No fine-tuned models configured")
        return False
    
    success_count = 0
    total_time = 0
    
    for model_name in finetuned_models:
        success, duration = await test_finetuned_model(model_name)
        if success:
            success_count += 1
            total_time += duration
    
    print(f"\n📊 Results: {success_count}/{len(finetuned_models)} models working")
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"⚡ Average response time: {avg_time:.2f}s")
    
    return success_count == len(finetuned_models)


async def test_parallel_generation():
    """Test parallel generation across multiple models."""
    print("\n⚡ Testing parallel generation across models...")
    
    finetuned_models = list(config.OLLAMA_MODEL_ENDPOINTS.keys())
    
    if len(finetuned_models) < 2:
        print("⚠️  Need at least 2 models for parallel testing")
        return
    
    # Use first 4 models (or all if less than 4)
    test_models = finetuned_models[:4]
    
    print(f"🔄 Testing parallel generation with {len(test_models)} models...")
    
    async def generate_from_model(model_name, prompt_suffix):
        model = Model(id=model_name, type="ollama")
        test_chat = Chat(messages=[
            ChatMessage(
                role=MessageRole.user, 
                content=f"Generate 3 numbers for sequence {prompt_suffix}: 100, 200, 300"
            )
        ])
        sample_cfg = SampleCfg(temperature=0.8)
        
        try:
            response = await sample(model, test_chat, sample_cfg)
            return model_name, response.completion
        except Exception as e:
            return model_name, f"Error: {e}"
    
    import time
    start_time = time.time()
    
    # Run all models in parallel
    tasks = [
        generate_from_model(model, f"#{i+1}")
        for i, model in enumerate(test_models)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    parallel_duration = end_time - start_time
    
    print(f"🏁 Parallel generation completed in {parallel_duration:.2f}s")
    print("📝 Results:")
    for model_name, result in results:
        print(f"  - {model_name}: {result[:50]}..." if len(result) > 50 else f"  - {model_name}: {result}")


async def benchmark_throughput():
    """Benchmark throughput of your fine-tuned models."""
    print("\n📈 Benchmarking model throughput...")
    
    # Use the first available fine-tuned model
    finetuned_models = list(config.OLLAMA_MODEL_ENDPOINTS.keys())
    
    if not finetuned_models:
        print("⚠️  No fine-tuned models available for benchmarking")
        return
    
    test_model = finetuned_models[0]
    print(f"🎯 Benchmarking with: {test_model}")
    
    model = Model(id=test_model, type="ollama")
    sample_cfg = SampleCfg(temperature=0.7)
    
    # Generate 5 small requests to test throughput
    requests = []
    for i in range(5):
        test_chat = Chat(messages=[
            ChatMessage(
                role=MessageRole.user, 
                content=f"Continue this sequence: {100+i*10}, {200+i*10}, {300+i*10}"
            )
        ])
        requests.append(test_chat)
    
    print("⏱️  Running 5 concurrent requests...")
    
    import time
    start_time = time.time()
    
    async def make_request(chat):
        return await sample(model, chat, sample_cfg)
    
    # Run requests concurrently
    tasks = [make_request(chat) for chat in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful_responses = [r for r in responses if not isinstance(r, Exception)]
    
    print(f"✅ {len(successful_responses)}/5 requests successful")
    print(f"⚡ Total time: {total_time:.2f}s")
    print(f"📊 Throughput: {len(successful_responses)/total_time:.2f} requests/second")


def print_setup_status():
    """Print current setup status and instructions."""
    print("\n" + "="*60)
    print("📋 FINE-TUNED MODEL SETUP STATUS")
    print("="*60)
    
    print("🔧 Configured model endpoints:")
    for model, endpoint in config.OLLAMA_MODEL_ENDPOINTS.items():
        print(f"  - {model} → {endpoint}")
    
    print(f"\n🌐 Default Ollama URL: {config.OLLAMA_BASE_URL}")
    
    print("\n💡 To start your fine-tuned models:")
    print("   bash start_ollama.sh")
    print("\n🧪 To test specific model:")
    print("   python test_finetuned_ollama.py")
    print("\n📊 To generate datasets:")
    print("   python scripts/generate_dataset_llama.py \\")
    print("       --config_module=cfgs/finetuned_models.py \\")
    print("       --cfg_var_name=toxic100_0_cfg \\")
    print("       --raw_dataset_path=./data/finetuned/raw.jsonl \\")
    print("       --filtered_dataset_path=./data/finetuned/filtered.jsonl")


if __name__ == "__main__":
    print("🦙 Testing fine-tuned Ollama models setup...")
    
    async def run_all_tests():
        # Check status of all endpoints
        running, endpoints = await test_multi_endpoint_status()
        
        if not running:
            print_setup_status()
            return
        
        # Test all fine-tuned models
        all_working = await test_all_finetuned_models()
        
        if all_working:
            # Run additional tests
            await test_parallel_generation()
            await benchmark_throughput()
            
            print("\n" + "="*50)
            print("🎉 All fine-tuned models are working perfectly!")
            print("🚀 Your multi-GPU Ollama setup is ready for research!")
            print("\n📋 Next steps:")
            print("1. Use cfgs/finetuned_models.py for your configurations")
            print("2. Generate datasets with your fine-tuned models")
            print("3. Compare results across different model variants")
        else:
            print("\n⚠️  Some models have issues. Check your start_ollama.sh setup.")
            print_setup_status()
    
    asyncio.run(run_all_tests()) 