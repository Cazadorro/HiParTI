//
// Created by Bolt on 12/15/2022.
//

#include "windows_compat.h"

#ifdef _MSC_VER
#include <chrono>
int vasprintf(char **strp, const char *format, va_list ap){
    int len = vscprintf(format, ap);
    if (len == -1)
        return -1;
    char *str = (char*)malloc((size_t) len + 1);
    if (!str)
        return -1;
    int retval = vsnprintf(str, len + 1, format, ap);
    if (retval == -1) {
        free(str);
        return -1;
    }
    *strp = str;
    return retval;
}


int asprintf(char **strp, const char *format, ...)
{
    va_list ap;
            va_start(ap, format);
    int retval = vasprintf(strp, format, ap);
            va_end(ap);
    return retval;
}

int clock_gettime(int clk_id, struct timespec *res) {
    auto time = std::chrono::steady_clock::now();
    auto secs = std::chrono::time_point_cast<std::chrono::seconds>(time);
    auto nano_secs = std::chrono::time_point_cast<std::chrono::nanoseconds>(time) - secs;
    res->tv_sec = secs.time_since_epoch().count();
    res->tv_nsec = nano_secs.count();
    //
//clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    return 0;
}

#endif