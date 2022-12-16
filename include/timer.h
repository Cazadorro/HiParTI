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
int gettimeofday(struct timeval* tp, struct timezone* tzp) {
    namespace sc = std::chrono;
    sc::system_clock::duration d = sc::system_clock::now().time_since_epoch();
    sc::seconds s = sc::duration_cast<sc::seconds>(d);
    tp->tv_sec = s.count();
    tp->tv_usec = sc::duration_cast<sc::microseconds>(d - s).count();
    return 0;
}
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


