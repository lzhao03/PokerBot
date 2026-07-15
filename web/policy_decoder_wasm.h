#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

uint8_t* poker_allocate(size_t size);
void poker_free(void* memory);
int poker_load_policy(const uint8_t* bytes, size_t size);
void poker_unload_policy();
int poker_strategy(uint32_t public_low,
                   uint32_t public_high,
                   uint32_t history,
                   uint32_t private_observation,
                   size_t action_count,
                   float* output);
uint32_t poker_model_low();
uint32_t poker_model_high();

}
