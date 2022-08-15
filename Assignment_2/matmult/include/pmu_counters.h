#ifndef __PMUCOUNTERS__
#define __PMUCOUNTERS__

unsigned int get_cyclecount(void);
void init_counters(int32_t do_reset, int32_t enable_divider);


#endif