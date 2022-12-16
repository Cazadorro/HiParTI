/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ctime>
#include <cstddef>
#include <cstdio>

#if defined(_WIN32)
#include <chrono>
#include <ctime>
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;
int gettimeofday(struct timeval* tp, struct timezone* tzp);
#else
#include <sys/time.h>
#endif // _WIN32

struct Timer{
  int running;
  double seconds;
  timeval Start;
  timeval Stop;
};


void timer_reset(Timer * const kTimer);

void timer_start(Timer * const kTimer);

void timer_stop(Timer * const kTimer);

void timer_fstart(Timer * const kTimer);


