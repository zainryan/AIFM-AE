#pragma once

void init_fm(void);
void flush_cache(void);
void dram_to_fm(void *dram, int64_t fm_idx, int64_t len);
void fm_to_dram(void *dram, int64_t fm_idx, int64_t len);
