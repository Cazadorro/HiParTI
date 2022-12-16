//
// Created by Bolt on 12/15/2022.
//

#ifndef HIPARTI_WINDOWS_COMPAT_H
#define HIPARTI_WINDOWS_COMPAT_H

#if defined(__GNUC__) && ! defined(_GNU_SOURCE)
#define _GNU_SOURCE /* needed for (v)asprintf, affects '#include <stdio.h>' */
#endif
#include <stdio.h>  /* needed for vsnprintf    */
#include <stdlib.h> /* needed for malloc, free */
#include <stdarg.h> /* needed for va_*         */

/*
 * vscprintf:
 * MSVC implements this as _vscprintf, thus we just 'symlink' it here
 * GNU-C-compatible compilers do not implement this, thus we implement it here
 */
#ifdef _MSC_VER
#define vscprintf _vscprintf
#endif

#ifdef __GNUC__
int vscprintf(const char *format, va_list ap)
{
    va_list ap_copy;
    va_copy(ap_copy, ap);
    int retval = vsnprintf(NULL, 0, format, ap_copy);
    va_end(ap_copy);
    return retval;
}
#endif

/*
 * asprintf, vasprintf:
 * MSVC does not implement these, thus we implement them here
 * GNU-C-compatible compilers implement these with the same names, thus we
 * don't have to do anything
 */
#ifdef _MSC_VER
#include <ctime>
int vasprintf(char **strp, const char *format, va_list ap);


int asprintf(char **strp, const char *format, ...);
#define CLOCK_MONOTONIC 1
int clock_gettime(int clk_id, std::timespec *res);

#define restrict __restrict
#endif

#endif //HIPARTI_WINDOWS_COMPAT_H
