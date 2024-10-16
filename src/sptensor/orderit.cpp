#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "timer.h"
#include "HiParTI.h"
#include "renumber.h"

#define TEST_CSR_ORDER_OUTPUT
#ifdef TEST_CSR_ORDER_OUTPUT
#include <iostream>
#endif
/** EXTRA INCLUDES**/
#include "hicoo_utils2.h"
#include <csrk.h>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <vector>
#include <span>
/** END EXTRA **/

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void orderDim(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const nm, ptiIndex * ndims, ptiIndex const dim, ptiIndex ** newIndices);

void orderforHiCOObfsLike(ptiIndex const nm, ptiNnzIndex const nnz, ptiIndex * ndims, ptiIndex ** coords, ptiIndex ** newIndices);

static double u_seconds(void)
{
    timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};
static void printCSR(ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols)
{
    ptiNnzIndex r, jend, jcol;
    printf("matrix of size %llu %u with %llu\n", m, n, ia[m+1]);
    
    for (r = 1; r <=m; r++)
    {
        jend = ia[r+1]-1;
        printf("r=%llu (%llu %llu)): ", r, ia[r], ia[r+1]);
        for(jcol = ia[r]; jcol <= jend ; jcol++)
            printf("%u ", cols[jcol]);
        printf("\n");
    }
}

static void checkRepeatIndex(ptiNnzIndex mtxNrows, ptiNnzIndex *rowPtrs, ptiIndex *cols, ptiIndex n )
{
    printf("\tChecking repeat indices\n");
    ptiIndex *marker = (ptiIndex *) calloc(n+1, sizeof(ptiIndex));
    ptiNnzIndex r,  jcol, jend;
    for (r = 1; r <= mtxNrows; r++)
    {
        jend = rowPtrs[r+1]-1;
        for (jcol = rowPtrs[r]; jcol <= jend; jcol++)
        {
            if( marker[cols[jcol]] < r )
                marker[cols[jcol]] = r;
            else if (marker[cols[jcol]] == r)
            {
                printf("*************************\n");
                printf("error duplicate col index %u at row %llu\n", cols[jcol], r);
                printf("*************************\n");
                
                exit(12);
            }
        }
        
    }
    free(marker);
}
static void checkEmptySlices(ptiIndex **coords, ptiNnzIndex nnz, ptiIndex nm, ptiIndex *ndims)
{
    ptiIndex m, i;
    ptiNnzIndex z;
    ptiIndex **marker;
    
    marker = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (ptiIndex*) calloc(ndims[m], sizeof(ptiIndex) );
    
    for (z = 0; z < nnz; z++)
        for (m=0; m < nm; m++)
            marker[m][coords[z][m]] = m + 1;
    
    for (m=0; m < nm; m++)
    {
        ptiIndex emptySlices = 0;
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                emptySlices ++;
        if(emptySlices)
            printf("dim %u, empty slices %u of %u\n", m, emptySlices,ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}

static void checkNewIndices(ptiIndex **newIndices, ptiIndex nm, ptiIndex *ndims)
{
    ptiIndex m, i;
    ptiIndex **marker, leftVoid;
    
    marker = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (ptiIndex*) calloc(ndims[m], sizeof(ptiIndex) );
    
    for (m=0; m < nm; m++)
        for (i = 0; i < ndims[m]; i++)
            marker[m][newIndices[m][i]] = m + 1;
    
    leftVoid = 0;
    for (m=0; m < nm; m++)
    {
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                leftVoid ++;
        if(leftVoid)
            printf("dim %u, left void %u of %u\n", m, leftVoid, ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}


void orderit(ptiSparseTensor * tsr, ptiIndex ** newIndices, int const renumber, ptiIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    ptiIndex i, m, nm = tsr->nmodes;
    ptiNnzIndex z, nnz = tsr->nnz;
    ptiIndex ** coords;
    ptiIndex its;
    
    /* copy the indices */
    ptiTimer copy_coord_timer;
    ptiNewTimer(&copy_coord_timer, 0);
    ptiStartTimer(copy_coord_timer);

    coords = (ptiIndex **) malloc(sizeof(ptiIndex*) * nnz);
    for (z = 0; z < nnz; z++)
    {
        coords[z] = (ptiIndex *) malloc(sizeof(ptiIndex) * nm);
        for (m = 0; m < nm; m++) {
            coords[z][m] = tsr->inds[m].data[z];
        }
    }
    for(z = 0; z < nnz; z++){
        fmt::println(" coords[{}] {},{},{}",z,
        coords[z][0],
                coords[z][1],
                coords[z][2]);
    }




    ptiStopTimer(copy_coord_timer);
    ptiPrintElapsedTime(copy_coord_timer, "Copy coordinate time");
    ptiFreeTimer(copy_coord_timer);
    
    /* checkEmptySlices(coords, nnz, nm, tsr->ndims); */

    if (renumber == 1) {    /* Lexi-order renumbering */

        ptiIndex ** orgIds = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);

        for (m = 0; m < nm; m++)
        {
            orgIds[m] = (ptiIndex *) malloc(sizeof(ptiIndex) * tsr->ndims[m]);
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("old.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nm; m++)
                orderDim(coords, nnz, nm, tsr->ndims, m, orgIds);
            
            // fprintf(stdout, "\niter %u:\n", its);
            // for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            //     ptiDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nm; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        for (m = 0; m < nm; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        orderforHiCOObfsLike(nm, nnz, tsr->ndims, coords, newIndices);
    }
    
    // printf("set the new indices\n");
/*    checkNewIndices(newIndices, nm, tsr->ndims);*/
    
    for (z = 0; z < nnz; z++)
        free(coords[z]);
    free(coords);
    
}
/******************** Internals begin ***********************/
/*beyond this line savages....
 **************************************************************/
static void printCoords(ptiIndex **coords, ptiNnzIndex nnz, ptiIndex nm)
{
    ptiNnzIndex z;
    ptiIndex m;
    for (z = 0; z < nnz; z++)
    {
        for (m=0; m < nm; m++)
            printf("%d ", coords[z][m]);
        printf("\n");
    }
}
/**************************************************************/
// static inline int isLessThanOrEqualToCoord(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
static inline int isLessThanOrEqualTo(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    ptiIndex m;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            if (z1[m] < z2[m])
                return -1;
            if (z1[m] > z2[m])
                return 1;
        }
    }
    return 0; /*are equal*/
}


static inline int isLessThanOrEqualToFast(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex * mode_order)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    ptiIndex i, m;
    
    for (i = 0; i < nm - 1; i ++)
    {
        m = mode_order[i];
        if (z1[m] < z2[m])
            return -1;
        if (z1[m] > z2[m])
            return 1;
    }
    return 0; /*are equal*/
}


static inline int isLessThanOrEqualToNewSum(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
// static inline int isLessThanOrEqualTo(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
{
    /*
     to sort the nonzeros first on i_1+i_2+...+i_4, if ties then on
     i_1+i_2+...+3, if ties then on i_1+i_2, if ties then on i_1 only.
     We do not include dim in the comparisons.
     
    */
    ptiIndex m;
    ptiIndex v1 = 0, v2 = 0;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            v1 += z1[m];
            v2 += z2[m];
        }
    }
    if(v1 < v2) return -1;
    else if(v1 > v2) return 1;
    else{
        for (m = 0; m < nm; m++)
        {
            if(m != dim)
            {
                v1 -= z1[m];
                v2 -= z2[m];
                if (v1 < v2) return -1;
                else if (v1 > v2) return 1;
            }
        }
    }
    return 0; /*are equal*/
}
/**************************************************************/
static inline void buSwap(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *wspace)
{
    ptiIndex m;
    
    for (m=0; m < nm; m++)
        wspace[m] = z2[m];
    
    for (m=0; m < nm; m++)
        z2[m] = z1[m];
    
    for (m=0; m < nm; m++)
        z1[m] = wspace[m];
    
}

static inline void writeInto(ptiIndex *target, ptiIndex *source, ptiIndex nm)
{
    ptiIndex m;
    for (m = 0; m < nm; m++)
        target[m] = source[m];
}

static void insertionSort(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    ptiNnzIndex z, z2plus;
    for (z = lo+1; z <= hi; z++)
    {
        writeInto(tmpNnz, coords[z], nm);
        /*find place for z*/
        z2plus = z;
        while ( z2plus > 0  && isLessThanOrEqualTo(coords[z2plus-1], tmpNnz, nm, ndims, dim)== 1)
        {
            writeInto(coords[z2plus], coords[z2plus-1], nm);
            z2plus --;
        }
        writeInto(coords[z2plus], tmpNnz, nm);
    }
}

static void insertionSortFast(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex * mode_order, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    ptiNnzIndex z, z2plus;
    for (z = lo+1; z <= hi; z++)
    {
        writeInto(tmpNnz, coords[z], nm);
        /*find place for z*/
        z2plus = z;
        while ( z2plus > 0  && isLessThanOrEqualToFast(coords[z2plus-1], tmpNnz, nm, mode_order)== 1)
        {
            writeInto(coords[z2plus], coords[z2plus-1], nm);
            z2plus --;
        }
        writeInto(coords[z2plus], tmpNnz, nm);
    }
}

static inline ptiNnzIndex buPartition(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    /* copied from the web http://ndevilla.free.fr/median/median/src/quickselect.c */
    ptiNnzIndex low, high, median, middle, ll, hh;
    
    low = lo; high = hi; median = (low+high)/2;
    for(;;)
    {
        if (high<=low) return median;
        if(high == low + 1)
        {
            if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim)== 1)
                buSwap (coords[high], coords[low], nm, wspace);
            return median;
        }
        middle = (low+high)/2;
        if(isLessThanOrEqualTo(coords[middle], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[middle], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[middle], coords[low], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[middle], nm, wspace);
        
        buSwap (coords[middle], coords[low+1], nm, wspace);
        
        ll = low + 1;
        hh = high;
        for (;;){
            do ll++; while (isLessThanOrEqualTo(coords[low], coords[ll], nm, ndims, dim) == 1);
            do hh--; while (isLessThanOrEqualTo(coords[hh], coords[low], nm, ndims, dim) == 1);
            
            if (hh < ll) break;
            
            buSwap (coords[ll], coords[hh], nm, wspace);
        }
        buSwap (coords[low], coords[hh], nm,wspace);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }
    
}

static inline ptiNnzIndex buPartitionFast(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex * mode_order, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    /* copied from the web http://ndevilla.free.fr/median/median/src/quickselect.c */
    ptiNnzIndex low, high, median, middle, ll, hh;
    
    low = lo; high = hi; median = (low+high)/2;
    for(;;)
    {
        if (high<=low) return median;
        if(high == low + 1)
        {
            if(isLessThanOrEqualToFast(coords[low], coords[high], nm, mode_order)== 1)
                buSwap (coords[high], coords[low], nm, wspace);
            return median;
        }
        middle = (low+high)/2;
        if(isLessThanOrEqualToFast(coords[middle], coords[high], nm, mode_order) == 1)
            buSwap (coords[middle], coords[high], nm, wspace);
        
        if(isLessThanOrEqualToFast(coords[low], coords[high], nm, mode_order) == 1)
            buSwap (coords[low], coords[high], nm, wspace);
        
        if(isLessThanOrEqualToFast(coords[middle], coords[low], nm, mode_order) == 1)
            buSwap (coords[low], coords[middle], nm, wspace);
        
        buSwap (coords[middle], coords[low+1], nm, wspace);
        
        ll = low + 1;
        hh = high;
        for (;;){
            do ll++; while (isLessThanOrEqualToFast(coords[low], coords[ll], nm, mode_order) == 1);
            do hh--; while (isLessThanOrEqualToFast(coords[hh], coords[low], nm, mode_order) == 1);
            
            if (hh < ll) break;
            
            buSwap (coords[ll], coords[hh], nm, wspace);
        }
        buSwap (coords[low], coords[hh], nm,wspace);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }   
    
}

/**************************************************************/
static void mySort(ptiIndex ** coords,  ptiNnzIndex last, ptiIndex nm, ptiIndex * ndims, ptiIndex dim)
{    
    /* sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /* an iterative quicksort */
    ptiNnzIndex *stack, top, lo, hi, pv;
    ptiIndex *tmpNnz, *wspace;
    
    tmpNnz = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    wspace = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    stack = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * 2 * (last+2));
    
    if(stack == NULL) {
        printf("could not allocated stack. returning\n");
        exit(14);
    }
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        pv = buPartition(coords, lo, hi, nm, ndims, dim, tmpNnz, wspace);
        
        if(pv > lo+1)
        {
            if(pv - lo > 128)
            {
                stack[top++] = lo;
                stack[top++] = pv-1 ;
            }
            else
                insertionSort(coords, lo, pv-1,  nm, ndims, dim, tmpNnz, wspace);
        }
        if(top >= 2 * (last+2)){
            printf("\thow come this tight?\n");
            exit(13);
        }
        if(pv + 1 < hi)
        {
            if(hi - pv > 128)
            {
                stack[top++] = pv + 1 ;
                stack[top++] = hi;
            }
            else
                insertionSort(coords, pv+1, hi,  nm, ndims, dim, tmpNnz, wspace);
        }
        if( top >= 2 * (last+2)) {
            printf("\thow come this tight?\n");
            exit(13);
        }
    }
    free(stack);
    free(wspace);
    free(tmpNnz);
}


static void mySortFast(ptiIndex ** coords,  ptiNnzIndex last, ptiIndex nm, ptiIndex * ndims, ptiIndex dim, ptiIndex * mode_order)
{
    /* sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /* an iterative quicksort */
    ptiNnzIndex *stack, top, lo, hi, pv;
    ptiIndex *tmpNnz, *wspace;
    
    tmpNnz = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    wspace = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    stack = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * 2 * (last+2));
    
    if(stack == NULL) {
        printf("could not allocated stack. returning\n");
        exit(14);
    }
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        pv = buPartitionFast(coords, lo, hi, nm, ndims, dim, mode_order, tmpNnz, wspace);
        
        if(pv > lo+1)
        {
            if(pv - lo > 128)
            {
                stack[top++] = lo;
                stack[top++] = pv-1 ;
            }
            else
                insertionSortFast(coords, lo, pv-1,  nm, ndims, dim, mode_order, tmpNnz, wspace);
        }
        if(top >= 2 * (last+2)){
            printf("\thow come this tight?\n");
            exit(13);
        }
        if(pv + 1 < hi)
        {
            if(hi - pv > 128)
            {
                stack[top++] = pv + 1 ;
                stack[top++] = hi;
            }
            else
                insertionSortFast(coords, pv+1, hi,  nm, ndims, dim, mode_order, tmpNnz, wspace);
        }
        if( top >= 2 * (last+2)) {
            printf("\thow come this tight?\n");
            exit(13);
        }
    }

    free(stack);
    free(wspace);
    free(tmpNnz);
}

static ptiIndex countNumItems(ptiIndex *setnext, ptiIndex *tailset, ptiIndex firstset, ptiIndex *prev)
{
    ptiIndex cnt = 0, set;
    for(set = firstset; set != 0; set = setnext[set])
    {
        ptiIndex item = tailset[set];
        
        while(item != 0 )
        {
            cnt ++;
            item = prev[item];
        }
    }
    return cnt;
}






void lexOrderThem(ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols, ptiIndex *cprm)
{
    /*m, n are the num of rows and cols, respectively. We lex order cols,
     given rows.    
     */

    ptiNnzIndex j, jcol, jend;
    ptiIndex jj;
    
    ptiIndex *freeIdList, freeIdTop;
    ptiIndex k, s, acol;
    ptiIndex firstset, set, pos, item, headset;
    
    colStruct *clms;
    setStruct *csets;
    clms = (colStruct *) calloc(sizeof(colStruct), n+2);
    csets = (setStruct *) calloc(sizeof(setStruct), n+2);

    freeIdList = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    
    initColDLL(clms, n);
    initSetDLL(csets,  n);

    for(jj = 1; jj<=n; jj++) {
        cprm[jj] = 2 * n;
    }
    
    firstset = 1;
    freeIdList[0] = 0;
    
    for(jj= 1; jj<=n; jj++) {
        freeIdList[jj] = jj + 1;/*1 is used as a set id*/
    }

    freeIdTop = 1;
    for(j=1; j<=m; j++){
        //for row (j) start to row (j) end
        jend = ia[j+1]-1;
        for(jcol = ia[j]; jcol <= jend ; jcol++){
            //iterate through column indices
            acol= cols[jcol];
            //grab the "column set index"
            s = clms[acol].svar;

            //if
            if( csets[s].flag < j){/*first occurence of supervar s in j*/            
                csets[s].flag = j;
                if(csets[s].sz == 1 && csets[s].tail != acol){
                    printf("this should not happen (sz 1 but tailset not ok)\n");
                    exit(12);
                }
                if(csets[s].sz > 1) {
                    ptiIndex newId;
                    /*remove acol from s*/
                    removeAColfromSet(csets, s, clms, acol);

                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    if(freeIdTop == n+1) {
                        printf("this should not happen (no index)\n");
                        exit(12);
                    }
                    newId = freeIdList[freeIdTop++];

                    appendAColtoSet(csets, newId, clms, acol);
                    csets[s].var = acol; /*the new set's important var is acol*/

                    insertSetBefore(csets, newId, s);/*newId is before s*/                    
                    if(firstset == s)
                        firstset = newId;
                    
                }
            }
            else{/*second or later occurence of s for row j*/
                k = csets[s].var;
                /*remove acol from its current chain*/               
                removeAColfromSet(csets, s, clms, acol);

                if(csets[s].sz == 0){/*s is a free id now..*/                
                    freeIdList[--freeIdTop] = s; /*add s to the free id list*/                    
                    setEmpty(csets, s);/*no need to adjust firstset, as this is the second occ of s*/
                }
                /*add to chain containing k (as the last element)*/
                appendAColtoSet(csets, clms[k].svar, clms, acol);
            }
        }
    }
    
    /*we are done. Let us read the cprm from the ordered sets*/
    pos = 1;
    for(set = firstset; set != 0; set = csets[set].next){
        item = csets[set].tail;
        headset = 0;
        
        while(item != 0 ){
            headset = item;
            item = clms[item].prev;
        }
        /*located the head of the set. output them (this is for keeping the order)*/
        while(headset){
            cprm[pos++] = headset;
            headset = clms[headset].next;
        }
    }
    
    free(freeIdList);
    free(csets);
    free(clms);
    
    if(pos-1 != n){
        printf("**************** Error ***********\n");
        printf("something went wrong and we could not order everyone\n");
        exit(12);
    }
    
    return ;
}


/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

// SB:
// coords is the SOA x,y,z... list of COO indexes
// nnz = number of non zeros
// nm = number of modes
// ndims = the length of each mode/dimension (length of x, y, and z)
// dim = the given dimension processing.
// orgIds = the origional order of all the IDs (same size as coords)
void orderDim(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const nm, ptiIndex * ndims, ptiIndex const dim, ptiIndex ** orgIds)
{
    ptiNnzIndex * rowPtrs=NULL, z, atRowPlus1, mtxNrows;
    ptiIndex * colIds=NULL, c;
    ptiIndex * cprm=NULL, * invcprm = NULL, * saveOrgIds;
    ptiNnzIndex mtrxNnz;

    //mode order that contains everything *except* our current mode.
    ptiIndex * mode_order = (ptiIndex *) malloc (sizeof(ptiIndex) * (nm - 1));
    ptiIndex i = 0;
    for(ptiIndex m = 0; m < nm; ++m) {
        if (m != dim) {
            mode_order[i] = m;
            ++ i;
        }
    }
    
    double t1, t0;
    t0 = u_seconds();
    // mySort(coords,  nnz-1, nm, ndims, dim);
    //SB:
    // sorts coords according to all dims except dim, where items are refered with newIndices
    // an iterative quicksort
    // appears to only be relevant to creating CSR version
    mySortFast(coords,  nnz-1, nm, ndims, dim, mode_order);
    t1 = u_seconds()-t0;
    printf("dim %u, sort time %.2f\n", dim, t1);
    // printCoords(coords, nnz, nm);
    /* we matricize this (others x thisDim), whose columns will be renumbered */
    
    /* on the matrix all arrays are from 1, and all indices are from 1. */
    
    rowPtrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nnz+2)); /*large space*/
    colIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (nnz+2)); /*large space*/
    
    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }
    
    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = coords[0][dim]+1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    //SB:
    //generates row pointers for CSR and increments col ids.
    //lexigraphically compares the indexes for the coordinates first.
    for (z = 1; z < nnz; z++)
    {
        // if(isLessThanOrEqualTo( coords[z], coords[z-1], nm, ndims, dim) != 0)
        //SB checking the other two dimensions, ignoring column.
        // not bothering with column order (could be any thing within a "row" (ie non dim dimensions)
        // could be that it doesn't matter? that you'd still have the same permutation regardless?

        if(isLessThanOrEqualToFast( coords[z], coords[z-1], nm, mode_order) != 0)
            rowPtrs[atRowPlus1 ++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz++] = coords[z][dim]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("dim %u create time %.2f\n", dim, t1);



    rowPtrs = reinterpret_cast<ptiNnzIndex*>(realloc(rowPtrs, (sizeof(ptiNnzIndex) * (mtxNrows+2))));

    for(std::size_t idx = 0; idx < mtxNrows+2; ++idx){
        fmt::println("row {} = {}", idx, rowPtrs[idx]);
    }
    for(std::size_t idx = 0; idx < nnz+2; ++idx){
        fmt::println("col {} = {}", idx, colIds[idx]);
    }

#ifdef TEST_CSR_ORDER_OUTPUT
    {
        std::cout << "row pointers:" << std::endl;
        for (std::size_t idx = 0; idx < (mtxNrows + 2); ++idx) {
            std::cout << rowPtrs[idx] << ", ";
        }
        std::cout << std::endl;
        std::cout << "nnz " << nnz << " mtxNrows " << mtxNrows << std::endl;
//        if ((mtxNrows + 2) == (nnz + 2)) {
//            std::cout << "nnz and mtxNrows match" << std::endl;
//        }
        std::cout << "colIds:" << std::endl;
        for (std::size_t idx = 0; idx < (nnz + 2); ++idx) {
            std::cout << colIds[idx] << ", ";
        }
        std::cout << std::endl;
    }
#endif
    cprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    invcprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    saveOrgIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    /*    checkRepeatIndex(mtxNrows, rowPtrs, colIds, ndims[dim] );*/

    // printf("rowPtrs: \n");
    // ptiDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // ptiDumpIndexArray(colIds, nnz + 2, stdout);
    
    t0 = u_seconds();
    lexOrderThem(mtxNrows, ndims[dim], rowPtrs, colIds, cprm);
    for(std::size_t idx = 0; idx < ndims[dim]+1; ++idx){
        std::cout << "cprm, " << idx << ", " << cprm[idx] << "\n";
    }
    t1 =u_seconds()-t0;
    printf("dim %u lexorder time %.2f\n", dim, t1);
    // printf("cprm: \n");
    // ptiDumpIndexArray(cprm, ndims[dim] + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < ndims[dim]; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[dim][c];
    }
    for (c=0; c < ndims[dim]; c++)
        orgIds[dim][c] = saveOrgIds[cprm[c+1]-1];
    
    // printf("invcprm: \n");
    // ptiDumpIndexArray(invcprm, ndims[dim] + 1, stdout);

    fmt::println("ALL OLD");
    for (z = 0; z < nnz; z++) {
        fmt::println(" coords[{}][{}] = {},{},{}", z, dim,
                     coords[z][0],
                     coords[z][1],
                     coords[z][2]);
    }
    fmt::println("OLD");
    for (z = 0; z < nnz; z++) {
        fmt::println(" coords[{}][{}] = {}", z, dim, coords[z][dim]);
    }

    /*rename the dim component of nonzeros*/
    fmt::println("NEW");
    for (z = 0; z < nnz; z++) {
        coords[z][dim] = invcprm[coords[z][dim]];
        fmt::println(" coords[{}][{}] = {}", z, dim, coords[z][dim]);
    }
    fmt::println("ALL NEW");
    for (z = 0; z < nnz; z++) {
        fmt::println(" coords[{}][{}] = {},{},{}", z, dim,
                     coords[z][0],
                     coords[z][1],
                     coords[z][2]);
    }
    free(mode_order);
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
}

/**************************************************************/

typedef struct{
    ptiIndex nvrt; /* number of vertices. This nvrt = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor
                   where the ith dimension is of size n_i.*/
    ptiNnzIndex *vptrs, *vHids; /*starts of hedges containing vertices, and the ids of the hedges*/
    
    ptiNnzIndex nhdg; /*this will be equal to the number of nonzeros in the tensor*/
    ptiNnzIndex *hptrs, *hVids; /*starts of vertices in the hedges, and the ids of the vertices*/
} basicHypergraph;

static void allocateHypergraphData(basicHypergraph *hg, ptiIndex nvrt, ptiNnzIndex nhdg, ptiNnzIndex npins)
{
    hg->nvrt = nvrt;
    hg->vptrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nvrt+1));
    hg->vHids = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * npins);
    
    hg->nhdg = nhdg;
    hg->hptrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nhdg+1));
    hg->hVids = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * npins);
}


static void freeHypergraphData(basicHypergraph *hg)
{
    hg->nvrt = 0;
    if (hg->vptrs) free(hg->vptrs);
    if (hg->vHids) free(hg->vHids);
    
    hg->nhdg = 0;
    if (hg->hptrs) free(hg->hptrs);
    if (hg->hVids) free(hg->hVids);
}


static void setVList(basicHypergraph *hg)
{
    /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
     hg->vptrs and hg->vHids are allocated appropriately.
     */
    
    ptiNnzIndex j, h, v, nhdg = hg->nhdg;
    
    ptiIndex nvrt = hg->nvrt;
    
    /*vertices */
    ptiNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids;
    /*hyperedges*/
    ptiNnzIndex *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    for (v = 0; v <= nvrt; v++)
        vptrs[v] = 0;
    
    for (h = 0; h < nhdg; h++)
    {
        for (j = hptrs[h]; j < hptrs[h+1]; j++)
        {
            v = hVids[j];
            vptrs[v] ++;
        }
    }
    for (v=1; v <= nvrt; v++)
        vptrs[v] += vptrs[v-1];
    
    for (h = nhdg; h >= 1; h--)
    {
        for (j = hptrs[h-1]; j < hptrs[h]; j++)
        {
            v = hVids[j];
            vHids[--(vptrs[v])] = h-1;
        }
    }
}

static void fillHypergraphFromCoo(basicHypergraph *hg, ptiIndex nm, ptiNnzIndex nnz, ptiIndex *ndims, ptiIndex **coords)
{
    
    ptiIndex  totalSizes;
    ptiNnzIndex h, toAddress;
    ptiIndex *dimSizesPrefixSum;
    
    ptiIndex i;
    
    dimSizesPrefixSum = (ptiIndex *) malloc(sizeof(ptiIndex) * (nm+1));
    totalSizes = 0;
    for (i=0; i < nm; i++)
    {
        dimSizesPrefixSum[i] = totalSizes;
        totalSizes += ndims[i];
    }
    printf("allocating hyp %u %llu\n", nm, nnz);
    
    allocateHypergraphData(hg, totalSizes, nnz, nnz * nm);
    
    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < nm; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + coords[h][i];
        toAddress += nm;
    }
    hg->hptrs[hg->nhdg] = toAddress;
    
    setVList(hg);
    free(dimSizesPrefixSum);
}
static inline ptiIndex locateVertex(ptiNnzIndex indStart, ptiNnzIndex indEnd, ptiNnzIndex *lst, ptiNnzIndex sz)
{
    ptiNnzIndex i;
    for (i = 0; i < sz; i++)
        if(lst[i] >= indStart && lst[i] <= indEnd)
            return lst[i];
    
    printf("could not locate in a hyperedge !!!\n");
    exit(1);
    return sz+1;
}

#define SIZEV( vid ) vptrs[(vid)+1]-vptrs[(vid)]
static void heapIncreaseKey(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex id, ptiIndex *inheap, ptiNnzIndex newKey)
{
    
    ptiIndex i = inheap[id]; /*location in heap*/
    if( i > 0 && i <=sz )
    {
        key[id] = newKey;
        
        while ((i>>1)>0 && ( (key[id] > key[heapIds[i>>1]]) ||
                            (key[id] == key[heapIds[i>>1]] && SIZEV(id) > SIZEV(heapIds[i>>1])))
               )
        {
            heapIds[i] = heapIds[i>>1];
            inheap[heapIds[i]] = i;
            i = i>>1;
        }
        heapIds[i] = id;
        inheap[id] = i;
    }
}


static void heapify(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex i,  ptiIndex *inheap)
{
    ptiIndex largest, j, l,r, tmp;
    
    largest = j = i;
    while(j<=sz/2)
    {
        l = 2*j;
        r = 2*j + 1;
        
        if ( (key[heapIds[l]] > key[heapIds[j]] ) ||
            (key[heapIds[l]] == key[heapIds[j]]  && SIZEV(heapIds[l]) < SIZEV(heapIds[j]) )
            )
            largest = l;
        else
            largest = j;
        
        if (r<=sz && (key[heapIds[r]]>key[heapIds[largest]] ||
                      (key[heapIds[r]]==key[heapIds[largest]] && SIZEV(heapIds[r]) < SIZEV(heapIds[largest])))
            )
            largest = r;
        
        if (largest != j)
        {
            tmp = heapIds[largest];
            heapIds[largest] = heapIds[j];
            inheap[heapIds[j]] = largest;
            
            heapIds[j] = tmp;
            inheap[heapIds[j]] = j;
            j = largest;
        }
        else
            break;
    }
}

static ptiIndex heapExtractMax(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex *sz, ptiIndex *inheap)
{
    ptiIndex maxind ;
    if (*sz < 1){
        printf("Error: heap underflow\n"); exit(12);
    }
    maxind = heapIds[1];
    heapIds[1] = heapIds[*sz];
    inheap[heapIds[1]] = 1;
    
    *sz = *sz - 1;
    inheap[maxind] = 0;
    
    heapify(heapIds, key, vptrs, *sz, 1, inheap);
    return maxind;
    
}

static void heapBuild(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex *inheap)
{
    ptiIndex i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}

static void orderforHiCOOaDim(basicHypergraph *hg, ptiIndex *newIndicesHg, ptiIndex indStart, ptiIndex indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/
    
    ptiIndex i, v, heapSz, *inHeap, *heapIds;
    ptiNnzIndex j, jj, hedge, hedge2, k, w, ww;
    ptiNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    ptiNnzIndex *keyvals, newKeyval;
    int *markers, mark;
    
    mark = 0;
    
    heapIds = (ptiIndex*) malloc(sizeof(ptiIndex) * (indEnd-indStart + 2));
    inHeap = (ptiIndex*) malloc(sizeof(ptiIndex) * hg->nvrt);/*this is large*/
    keyvals = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * hg->nvrt);
    markers = (int*) malloc(sizeof(int)* hg->nvrt);
    
    heapSz = 0;
    
    for (i = indStart; i<=indEnd; i++)
    {
        keyvals[i] = 0;
        heapIds[++heapSz] = i;
        inHeap[i] = heapSz;
        markers[i] = -1;
    }
    heapBuild(heapIds, keyvals, vptrs, heapSz, inHeap);
    
    for (i = indStart; i <= indEnd; i++)
    {
        v = heapExtractMax(heapIds, keyvals, vptrs, &heapSz, inHeap);
        newIndicesHg[v] = i;
        markers[v] = mark;
        for (j = vptrs[v]; j < vptrs[v+1]; j++)
        {
            hedge = vHids[j];
            for (k = hptrs[hedge]; k < hptrs[hedge+1]; k++)
            {
                w = hVids[k];
                if (markers[w] != mark)
                {
                    markers[w] = mark;
                    for(jj = vptrs[w]; jj < vptrs[w+1]; jj++)
                    {
                        hedge2 = vHids[jj];
                        ww = locateVertex(indStart, indEnd, hVids + hptrs[hedge2], hptrs[hedge2+1]-hptrs[hedge2]);
                        if( inHeap[ww] )
                        {
                            newKeyval = keyvals[ww] + 1;
                            heapIncreaseKey(heapIds, keyvals, vptrs, heapSz, ww, inHeap, newKeyval);
                        }
                    }
                }
            }
        }
    }
    
    free(markers);
    free(keyvals);
    free(inHeap);
    free(heapIds);
}

struct BasicHypergraph{
    /*starts of hedges containing vertices, and the ids of the hyper edges*/
    std::vector<ptiNnzIndex> vertex_ptrs;
    std::vector<ptiNnzIndex> vertex_hyper_edge_ids;

    /*starts of vertices in the hedges, and the ids of the vertices*/
    std::vector<ptiNnzIndex> hyper_edge_ptrs;
    std::vector<ptiNnzIndex> hyper_edge_vertex_ids;


    BasicHypergraph() = default;
    /* vertex_count. This vertex_count = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor where the ith dimension is of size n_i.*/
    /* hyper_edge_count will be equal to the number of nonzeros in the tensor*/
    /* starts of vertices in the hedges, and the ids of the vertices */
    BasicHypergraph(ptiIndex vertex_count, ptiNnzIndex hyper_edge_count, ptiNnzIndex id_count) :
            vertex_ptrs(vertex_count + 1),
            vertex_hyper_edge_ids(id_count),
            hyper_edge_ptrs(hyper_edge_count + 1),
            hyper_edge_vertex_ids(id_count) {

    }
    [[nodiscard]]
    ptiIndex vertex_count() const{
        return vertex_ptrs.size() - 1;
    }
    [[nodiscard]]
    ptiNnzIndex hyper_edge_count() const{
        return hyper_edge_ptrs.size() - 1;
    }

    void set_vertex_list(){
        /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
         hg->vptrs and hg->vHids are allocated appropriately.
        */
        std::fill(vertex_ptrs.begin(), vertex_ptrs.end(), ptiNnzIndex(0));

        for (std::size_t hedge_idx = 0; hedge_idx < hyper_edge_count(); ++hedge_idx){
            for (std::size_t i = hyper_edge_ptrs[hedge_idx]; i < hyper_edge_ptrs[hedge_idx+1]; i++){
                auto v = hyper_edge_vertex_ids[i];
                vertex_ptrs[v]++;
            }
        }
        //Prefix summing...?
        for (std::size_t vertex_idx=1; vertex_idx <= vertex_count(); vertex_idx++) {
            vertex_ptrs[vertex_idx] += vertex_ptrs[vertex_idx - 1];
        }

        //goind reverse.
        for (std::size_t hedge_idx = hyper_edge_count(); hedge_idx >= 1; hedge_idx--){
            for (std::size_t i = hyper_edge_ptrs[hedge_idx-1]; i < hyper_edge_ptrs[hedge_idx]; i++){
                auto v = hyper_edge_vertex_ids[i];
                //reeeeally not sure why this being done, I guess it should still provide the mapping between vertex->hyper edge anyway?
                vertex_hyper_edge_ids[--(vertex_ptrs[v])] = hedge_idx-1;
            }
        }
    }

    [[nodiscard]]
    static BasicHypergraph from_coo(ptiIndex mode_count, ptiNnzIndex nnz, std::span<const ptiIndex> mode_sizes, ptiIndex **coords){

        BasicHypergraph hg;

        ptiIndex  totalSizes;
        ptiNnzIndex h;
        //TODO why nm + 1?
        std::vector<ptiIndex> mode_size_prefix_sum(mode_count+1, 0);

        ptiIndex i;

        for (std::size_t idx  = 1; idx < mode_count; idx++){
            mode_size_prefix_sum[idx] = mode_sizes[idx-1] + mode_size_prefix_sum[idx-1];
        }
        //the last value of the prefix sum, plus the last value of the mode_sizes ends up being the total
        //sum of all mode sizes summed up.
        ptiIndex total_mode_sum = mode_size_prefix_sum.back() + mode_sizes.back();

        printf("allocating hyp %u %llu\n", mode_count, nnz);

        hg = BasicHypergraph(total_mode_sum, nnz, nnz * mode_count);

        ptiNnzIndex toAddress = 0;
        for (h = 0; h < nnz; h++){
            hg.hyper_edge_ptrs[h] = toAddress;
            for (i = 0;  i < mode_count; i++)
                hg.hyper_edge_vertex_ids[toAddress + i] = mode_size_prefix_sum[i] + coords[h][i];
            toAddress += mode_count;
        }
        hg.hyper_edge_ptrs[hg.hyper_edge_count()] = toAddress;

        hg.set_vertex_list();
        return hg;
    }
};


static void heapifyTranslated(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex i,  ptiIndex *inheap)
{
    ptiIndex largest, j, l,r, tmp;

    largest = j = i;
    while(j<=sz/2)
    {
        l = 2*j;
        r = 2*j + 1;

        if ( (key[heapIds[l]] > key[heapIds[j]] ) ||
             (key[heapIds[l]] == key[heapIds[j]]  && SIZEV(heapIds[l]) < SIZEV(heapIds[j]) )
                )
            largest = l;
        else
            largest = j;

        if (r<=sz && (key[heapIds[r]]>key[heapIds[largest]] ||
                      (key[heapIds[r]]==key[heapIds[largest]] && SIZEV(heapIds[r]) < SIZEV(heapIds[largest])))
                )
            largest = r;

        if (largest != j)
        {
            tmp = heapIds[largest];
            heapIds[largest] = heapIds[j];
            inheap[heapIds[j]] = largest;

            heapIds[j] = tmp;
            inheap[heapIds[j]] = j;
            j = largest;
        }
        else
            break;
    }
}

static void heapBuildTranslated(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex *inheap)
{
    ptiIndex i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}
static void orderforHiCOOaDimTranslated(basicHypergraph *hg, ptiIndex *newIndicesHg, ptiIndex indStart, ptiIndex indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/

    ptiIndex i, v, heapSz, *inHeap, *heapIds;
    ptiNnzIndex j, jj, hedge, hedge2, k, w, ww;
    ptiNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;

    ptiNnzIndex *keyvals, newKeyval;
    int *markers, mark;

    mark = 0;

    heapIds = (ptiIndex*) malloc(sizeof(ptiIndex) * (indEnd-indStart + 2));
    inHeap = (ptiIndex*) malloc(sizeof(ptiIndex) * hg->nvrt);/*this is large*/
    keyvals = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * hg->nvrt);
    markers = (int*) malloc(sizeof(int)* hg->nvrt);

    heapSz = 0;

    for (i = indStart; i<=indEnd; i++)
    {
        keyvals[i] = 0;
        heapIds[++heapSz] = i;
        inHeap[i] = heapSz;
        markers[i] = -1;
    }
    heapBuildTranslated(heapIds, keyvals, vptrs, heapSz, inHeap);

    for (i = indStart; i <= indEnd; i++)
    {
        v = heapExtractMax(heapIds, keyvals, vptrs, &heapSz, inHeap);
        newIndicesHg[v] = i;
        markers[v] = mark;
        for (j = vptrs[v]; j < vptrs[v+1]; j++)
        {
            hedge = vHids[j];
            //find all hyper edges connected to the vertex removed
            for (k = hptrs[hedge]; k < hptrs[hedge+1]; k++)
            {
                //for every vertex connected to that hyperedge (breadth first search???)
                w = hVids[k];
                //if we haven't already removed that vertex.
                if (markers[w] != mark)
                {
                    //mark the vertex
                    markers[w] = mark;

                    for(jj = vptrs[w]; jj < vptrs[w+1]; jj++)
                    {
                        //for every hyper edge connected to *that* vertex.
                        hedge2 = vHids[jj];
                        //find first vertex that's in current dimension (indstart indend?)
                        ww = locateVertex(indStart, indEnd, hVids + hptrs[ hedge2], hptrs[hedge2+1]-hptrs[hedge2]);
                        // if it's in heap
                        if( inHeap[ww] )
                        {
                            //increase the key?
                            newKeyval = keyvals[ww] + 1;
                            heapIncreaseKey(heapIds, keyvals, vptrs, heapSz, ww, inHeap, newKeyval);
                        }
                    }
                }
            }
        }
    }

    free(markers);
    free(keyvals);
    free(inHeap);
    free(heapIds);
}


//void orderforHiCOObfsLikeTranslated(ptiIndex const mode_count, ptiNnzIndex const nnz, std::span<const ptiIndex> mode_sizes, ptiIndex ** coords, ptiIndex ** newIndices){
//    /*PRE: newIndices is allocated
//
//     POST:
//     newIndices[0][0...n_0-1] gives the new ids for dim 0
//     newIndices[1][0...n_1-1] gives the new ids for dim 1
//     ...
//     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
//
//     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
//     */
//    ptiIndex d, i;
//    std::vector<ptiIndex> mode_size_prefix_sum(mode_count, 0);
//
//    for (std::size_t idx  = 1; idx < mode_count; idx++){
//        mode_size_prefix_sum[idx] = mode_sizes[idx-1] + mode_size_prefix_sum[idx-1];
//    }
//
//
//    auto hg = BasicHypergraph::from_coo(mode_count, nnz, mode_sizes, coords);
//
//    std::vector<ptiIndex> new_indices_hyper_graph(hg.vertex_count());
//
//    //incrementing 0 -> ... max.
//    std::iota(new_indices_hyper_graph.begin(), new_indices_hyper_graph.end(), 0);
//
//
//    for (d = 0; d < nm; d++) /*order d*/
//        //believe we prefix sum in order to index the appropriate set of vertices?
//        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
//
//    /*copy from newIndices to newIndicesOut*/
//    for (d = 0; d < nm; d++)
//        for (i = 0; i < ndims[d]; i++)
//            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
//
//    free(newIndicesHg);
//    freeHypergraphData(&hg);
//    free(dimsPrefixSum);
//
//}

/**************************************************************/
void orderforHiCOObfsLike(ptiIndex const nm, ptiNnzIndex const nnz, ptiIndex * ndims, ptiIndex ** coords, ptiIndex ** newIndices)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    ptiIndex d, i;
    ptiIndex *dimsPrefixSum;
    
    basicHypergraph hg;
    
    ptiIndex *newIndicesHg;
    
    dimsPrefixSum = (ptiIndex*) calloc(nm, sizeof(ptiIndex));
    for (d = 1; d < nm; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, nm,  nnz, ndims, coords);
    newIndicesHg = (ptiIndex*) malloc(sizeof(ptiIndex) * hg.nvrt);
    
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < nm; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nm; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
/********************** Internals end *************************/


/** EXTRA **/
std::vector<std::uint32_t> find_column_permutation(
        std::span<const std::uint32_t> old_r_vec, std::span<const std::uint32_t> old_c_vec,
        std::span<const unsigned int> new_r_vec, std::span<const unsigned int> new_c_vec,
        std::span<const unsigned int> r_vec_perm, unsigned int dim_length){

    assert(old_c_vec.size() == new_c_vec.size());
    assert(old_r_vec.size() == new_r_vec.size());
    assert(r_vec_perm.size() == new_r_vec.size());
    auto row_count = old_r_vec.size();
    auto col_count = old_c_vec.size();
    std::vector<std::uint32_t> mapping(dim_length);
    //currently will maintain empty column order, but maybe better not to do that, and shove empty to the end? Seems to
    // have no practical consequence, since empty isn't stored anyway?
    std::iota(mapping.begin(), mapping.end(), 0);

    //every mapping is assumed to be in order, until changed.
    for(std::size_t i = 0; i < row_count - 1; ++i){
        auto old_start = old_r_vec[r_vec_perm[i]];
        auto old_end = old_r_vec[r_vec_perm[i] + 1];
        auto old_range = std::span(old_c_vec.begin() + old_start, old_c_vec.begin() + old_end);

        auto new_start = new_r_vec[i];
        auto new_end = new_r_vec[i + 1];
        auto new_range = std::span(old_c_vec.begin() + new_start, old_c_vec.begin() + new_end);

        assert(old_range.size() == new_range.size());

        auto row_element_size = old_range.size();

        //every element in new columns should map to old columns
        //we do extra work here, but gaurantees changes to columns are properly mapped here.
        for(std::size_t c = 0; c < row_element_size; ++c){
            auto old_col_val = old_range[c];
            auto new_col_val = new_range[c];
            mapping[old_col_val] = new_col_val;
        }
    }
    return mapping;
}

//calculates the row given a set of indices for a value, tne number of modes, the size of each mode,
// and the list of mod indexes that make up the given row.
ptiIndex calc_row(const ptiIndex *z1, ptiIndex nm, const ptiIndex * ndims, const ptiIndex * mode_order){
    /*is z1 less than or equal to z2 for all indices except dim?*/
    ptiIndex row_index = 0;
    std::size_t accumulated_dim = 1;
    for (ptiIndex i = 0; i < nm - 1; i ++){
        ptiIndex m = mode_order[i];
        row_index += z1[m] * accumulated_dim;
        accumulated_dim *= ndims[m];
    }
    return row_index;
}

ptiIndex calc_col(const ptiIndex *z1, ptiIndex dim){
    /*is z1 less than or equal to z2 for all indices except dim?*/
    return z1[dim];
}


//sorts smallest to largest.
void sortFrames(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex mode_count, ptiIndex col_mode, ptiIndex row_mode){
    std::vector<std::size_t> indexes(nnz);
    std::iota(indexes.begin(), indexes.end(), 0);
    if(mode_count <= 2){
        return;
    }

    //sort just the indices. +
    std::sort(indexes.begin(), indexes.end(), [&](auto lhs, auto rhs){

        for(std::size_t mode = 0; mode < mode_count; ++mode){
            if(mode != col_mode && mode != row_mode){
                if(coords[lhs][mode] < coords[rhs][mode]){
                    return true;
                } else if(coords[lhs][mode] > coords[rhs][mode]){
                    return false;
                }
            }
        }
        //process rows *after* should maintain, effectively emulating [frames][rows][cols] ordering.
        if(coords[lhs][row_mode] < coords[rhs][row_mode]){
            return true;
        } else if(coords[lhs][row_mode] > coords[rhs][row_mode]){
            return false;
        }
        //finally sort by cols.
        if(coords[lhs][col_mode] < coords[rhs][col_mode]){
            return true;
        } else if(coords[lhs][col_mode] > coords[rhs][col_mode]){
            return false;
        }
        return false;
    });

    //copy result back into coords.
    std::vector<ptiIndex> copy(nnz);
    for(std::size_t mode = 0; mode < mode_count; ++mode){
        for(std::size_t i = 0; i < nnz; ++i){
            copy[i] = coords[indexes[i]][mode];
        }
        for(std::size_t i = 0; i < nnz; ++i){
            coords[i][mode] = copy[i];
        }
    }
}

//coords = [xyz,xyz,xyz.... nnz = number of non zeros.  nm = number of modes. ndims = size of each mode. dim = chosen dim. orgIds = permutation change.
void orderBandK2(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const mode_count, ptiIndex * mode_sizes, ptiIndex const chosen_mode, ptiIndex ** orgIds) {

    assert((mode_count != 1, "Currently expects mode count to be more than 1"));
    ptiIndex col_mode = chosen_mode;
    ptiIndex row_mode = (chosen_mode + 1) % mode_count;
    ptiIndex col_size = mode_sizes[col_mode];
    ptiIndex row_size = mode_sizes[row_mode];
    //other modes non col.
    std::vector<ptiIndex> other_modes;
    ptiIndex frame_count = 1;
    other_modes.reserve(mode_count - 2);
    for (std::size_t mode = 0; mode < mode_count; ++mode) {
        if (mode != col_mode && mode != row_mode) {
            other_modes.push_back(mode);
            frame_count *= mode_sizes[mode];
        }
    }
    sortFrames(coords, nnz, mode_count, col_mode, row_mode);


    auto extract_frame_indexes = [&](std::size_t i) {
        std::vector<ptiIndex> frame_indexes;
        frame_indexes.reserve(mode_count - 2);
        for (std::size_t mode = 0; mode < mode_count; ++mode) {
            if (mode != col_mode && mode != row_mode) {
                frame_indexes.push_back(coords[i][mode]);
            }
        }
        return frame_indexes;
    };
    //iterating through each sorted by frames, then rows, the cols.
    std::vector<ptiIndex> previous_frame_indexes(mode_count - 2);
    previous_frame_indexes = extract_frame_indexes(0);
    util::Transpose2DBitfield transpose_bitfield(std::max(row_size, col_size));
    std::size_t transpose_nnz = 0;
    std::vector<std::size_t> prev_column_permutation(mode_sizes[col_mode]);
    std::vector<std::size_t> next_column_permutation(mode_sizes[col_mode]);
    auto permute_frame = [&](){
        //TODO other_mode_size is the "row" size, but will need column size as well and get max for square, for now assuming square.
        std::vector<std::uint32_t> row_ptrs_full(transpose_bitfield.width() + 1);
        std::vector<std::uint32_t> col_ids_full(transpose_nnz);
        row_ptrs_full[0] = 0;
        std::size_t row_idx = 0;
        col_ids_full[0] = 0; //TODO not sure the point of this one shouldn't need to do anything.
        std::size_t last_row = 0;

        std::size_t accumulated_idx = 0;
        for (std::size_t row = 0; row < transpose_bitfield.width(); ++row) {
            for (std::size_t col = 0; col < transpose_bitfield.width(); ++col) {
                if (transpose_bitfield.get(row, col)) {
                    col_ids_full[accumulated_idx] = col;
                    accumulated_idx += 1;
                }
            }
            row_ptrs_full[row + 1] = accumulated_idx;
        }
        const char *kernelType = "SpMV";
        const char *corseningType = "HAND";
        const char *orderingType = "";
        int k = 2;
        std::vector<int> supRowSizes = {1};

        std::vector<ptiValue> values(transpose_nnz + 1, 1.0f);
        CSRk_Graph A_mat(transpose_bitfield.width(), transpose_bitfield.width(), transpose_nnz,
                         row_ptrs_full.data(), col_ids_full.data(), values.data(), kernelType,
                         orderingType, corseningType, false, k, supRowSizes.data());

        A_mat.putInCSRkFormat();

        auto row_perm_span = std::span(A_mat.getPermutation(), row_size);

        //update previous permutations.
        assert(row_perm_span.size() == next_column_permutation.size());
        for (std::size_t perm_idx = 0; perm_idx < row_perm_span.size(); ++perm_idx) {
            next_column_permutation[perm_idx] = prev_column_permutation[row_perm_span[perm_idx]];
        }
        prev_column_permutation = next_column_permutation;
        transpose_bitfield.clear();
        transpose_nnz = 0;
    };
    for (std::size_t i = 0; i < nnz; ++i) {
        auto current_frame_indexes = extract_frame_indexes(i);
        if (current_frame_indexes == previous_frame_indexes) {
            auto curr_row = coords[i][row_mode];
            auto curr_col = coords[i][col_mode];
            auto already_set = transpose_bitfield.test_and_set(curr_row, curr_col);
            if (!already_set) {
                transpose_nnz += 1;
            }
        } else {
           permute_frame();
           previous_frame_indexes = current_frame_indexes;
        }
    }
    //won't trigger last iteration with last frame, as difference won't be found.
     permute_frame();

    std::vector<std::uint32_t> cprm(prev_column_permutation.begin(), prev_column_permutation.end());

    //need to move it *back* into being 1 based.
    for(auto& value : cprm){
        value += 1;
    }
    //suppposed to be 1 larger.
    cprm.insert(cprm.begin(), 0);

    auto invcprm = std::vector<ptiIndex>(mode_sizes[chosen_mode]+1);
    auto saveOrgIds = std::vector<ptiIndex>(mode_sizes[chosen_mode]+1);

    /* update orgIds and modify coords */
    for (std::size_t c=0; c < mode_sizes[chosen_mode]; c++){
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[chosen_mode][c];
    }
    for (std::size_t c=0; c < mode_sizes[chosen_mode]; c++) {
        orgIds[chosen_mode][c] = saveOrgIds[cprm[c + 1] - 1];
    }
    /*rename the dim component of nonzeros*/
    for (std::size_t z = 0; z < nnz; z++) {
        coords[z][chosen_mode] = invcprm[coords[z][chosen_mode]];
    }
}

void orderBandK(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const nm, ptiIndex * ndims, ptiIndex const dim, ptiIndex ** orgIds)
{
    using nnzIndexType = std::uint32_t;
    nnzIndexType * rowPtrs=NULL, z, atRowPlus1, mtxNrows;
    ptiIndex * colIds=NULL, c;
    ptiIndex * invcprm = NULL, * saveOrgIds;
    nnzIndexType mtrxNnz;


    ptiIndex * mode_order = (ptiIndex *) malloc (sizeof(ptiIndex) * (nm - 1));
    //needed to get the "row" size.
    std::size_t other_mode_size = 1;
    ptiIndex i = 0;
    for(ptiIndex m = 0; m < nm; ++m) {
        if (m != dim) {
            mode_order[i] = m;
            //accumulate the "row" size from all other dims except the dim that corresponds to column.
            other_mode_size *= ndims[m];
            ++ i;
        }
    }

    double t1, t0;
    t0 = u_seconds();
    // mySort(coords,  nnz-1, nm, ndims, dim);
    //SB:
    // sorts coords according to all dims except dim, where items are refered with newIndices
    // an iterative quicksort
    // appears to only be relevant to creating CSR version
    mySortFast(coords,  nnz-1, nm, ndims, dim, mode_order);
    t1 = u_seconds()-t0;
    printf("dim %u, sort time %.2f\n", dim, t1);
    // printCoords(coords, nnz, nm);
    /* we matricize this (others x thisDim), whose columns will be renumbered */

    /* on the matrix all arrays are from 1, and all indices are from 1. */

    rowPtrs = (nnzIndexType *) malloc(sizeof(nnzIndexType ) * (nnz+2)); /*large space*/
    colIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (nnz+2)); /*large space*/

    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }

    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = coords[0][dim]+1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */

    t0 = u_seconds();



    //TODO other_mode_size is the "row" size, but will need column size as well and get max for square, for now assuming square.
    util::Transpose2DBitfield transpose_bitfield(std::max(other_mode_size, static_cast<std::size_t>(ndims[dim])));
    auto transpose_nnz = 0;
    for(std::size_t idx = 0; idx < nnz; ++idx){
        auto curr_row = calc_row(coords[idx], nm, ndims, mode_order);
        auto curr_col = calc_col(coords[idx], dim);
        auto already_set = transpose_bitfield.test_and_set(curr_row, curr_col);
        if(!already_set){
            transpose_nnz += 1;
        }
    }

    //SB:
    //generates row pointers for CSR and increments col ids.
    //lexigraphically compares the indexes for the coordinates first.

    //row pointers always one more than number of rows to encompass last element.
//    std::vector<std::uint32_t> row_ptrs_full(other_mode_size + 1);
    std::vector<std::uint32_t> row_ptrs_full(transpose_bitfield.width() + 1);
    std::vector<std::uint32_t>  col_ids_full(transpose_nnz);
    row_ptrs_full[0] = 0;
    std::size_t row_idx = 0;
    col_ids_full[0] = coords[0][dim];
    std::size_t last_row = 0;

    std::size_t accumulated_idx = 0;
    for(std::size_t row = 0; row < transpose_bitfield.width(); ++row){
        for(std::size_t col = 0; col < transpose_bitfield.width(); ++col){
            if(transpose_bitfield.get(row, col)){
                col_ids_full[accumulated_idx] = col;
                accumulated_idx+=1;
            }
        }
        row_ptrs_full[row + 1] = accumulated_idx;
    }

//    for(std::size_t idx = 1; idx < nnz; ++idx){
//        auto curr_row = calc_row(coords[idx], nm, ndims, mode_order);
//        assert(curr_row >= last_row);
//        if(curr_row > last_row){
//            for(std::size_t row = last_row + 1; row <= curr_row; ++row){
//                row_ptrs_full[row] = idx;
//            }
//            last_row = curr_row;
//        }
//        col_ids_full[idx] = coords[idx][dim];
//    }
//    if(last_row < (row_ptrs_full.size() - 1)){
//        for(std::size_t row = last_row + 1; row < row_ptrs_full.size(); ++row){
//            row_ptrs_full[row] = nnz;
//        }
//    }

    for (z = 1; z < nnz; z++)
    {
        // if(isLessThanOrEqualTo( coords[z], coords[z-1], nm, ndims, dim) != 0)
        //SB checking the other two dimensions, ignoring column.
        // not bothering with column order (could be any thing within a "row" (ie non dim dimensions)
        // could be that it doesn't matter? that you'd still have the same permutation regardless?

        if(isLessThanOrEqualToFast( coords[z], coords[z-1], nm, mode_order) != 0)
            rowPtrs[atRowPlus1 ++] = mtrxNnz; /* close the previous row and start a new one. */

        colIds[mtrxNnz++] = coords[z][dim]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("dim %u create time %.2f\n", dim, t1);
    //reserved space, now reallocating smaller.
    rowPtrs = reinterpret_cast<nnzIndexType *>(realloc(rowPtrs, (sizeof(nnzIndexType ) * (mtxNrows+2))));
    auto row_ptrs_view = std::span(rowPtrs, mtxNrows+2);
#ifdef TEST_CSR_ORDER_OUTPUT
    {
        std::cout << "row pointers:" << std::endl;
        for (std::size_t idx = 0; idx < (mtxNrows + 2); ++idx) {
            std::cout << rowPtrs[idx] << ", ";
        }
        std::cout << std::endl;
        std::cout << "nnz " << nnz << " mtxNrows " << mtxNrows << std::endl;
//        if ((mtxNrows + 2) == (nnz + 2)) {
//            std::cout << "nnz and mtxNrows match" << std::endl;
//        }
        std::cout << "colIds:" << std::endl;
        for (std::size_t idx = 0; idx < (nnz + 2); ++idx) {
            std::cout << colIds[idx] << ", ";
        }
        std::cout << std::endl;
    }
#endif
    //cprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    invcprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    saveOrgIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    /*    checkRepeatIndex(mtxNrows, rowPtrs, colIds, ndims[dim] );*/

    // printf("rowPtrs: \n");
    // ptiDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // ptiDumpIndexArray(colIds, nnz + 2, stdout);


    const char * kernelType = "SpMV";
    const char * corseningType = "HAND";
    const char * orderingType = "";
    int k = 2;
    std::vector<int> supRowSizes = {1};
    // TODO coords is actually just a set of [[x,y,z],[x,y,z]...]
    //TODO
    // input is COO
    // at this point it has been converted to CSR (row pointers and col ids)
    // so basically all we want to do is just modify the part for getting the permutation out
    //TODO probably don't actually want real values to be here?

    //TODO make sure don't need to worry about off by 1 for colIDs?
    //TODO can we assume colIds are sorted amoung themselves already? The sort function doesn't actually sort
    // current dim, just with respect to everything else.

    //Do we have to invert the logic for row and col ids in order to properly get cols (as rows?)
    auto row_size = mtxNrows+2;
//    std::vector<ptiValue> values(nnz + 1, 1.0f);
    std::vector<ptiValue> values(transpose_nnz + 1, 1.0f);
    //(we don't want to access first thing, so we don't touch row_ptrs)

    //TODO if need to move input to be zero based.
    std::vector<std::uint32_t> old_row_ptrs(row_size);
    std::vector<std::uint32_t> old_col_ids(nnz);
    for(std::size_t idx = 0; idx < old_row_ptrs.size(); ++idx){
        old_row_ptrs[idx] = rowPtrs[idx + 1] - 1;
    }
//    old_row_ptrs[0] = 0;
    for(std::size_t idx = 0; idx < old_col_ids.size(); ++idx){
        old_col_ids[idx] = colIds[idx+1] - 1;
    }

//    std::vector<std::uint32_t> old_row_ptrs(row_size);
//    std::vector<std::uint32_t> old_col_ids(nnz + 1);
//    for(std::size_t i = 0; i < old_row_ptrs.size(); ++i){
//        old_row_ptrs[i] = rowPtrs[i];
//    }
//    for(std::size_t i = 0; i < old_col_ids.size(); ++i){
//        old_col_ids[i] = colIds[i];
//    }

//    CSRk_Graph A_mat(mtxNrows + 1, ndims[dim], nnz, old_row_ptrs.data(), old_col_ids.data(), values.data(), kernelType,
//                     orderingType, corseningType, false, k, supRowSizes.data());

//    CSRk_Graph A_mat(ndims[dim], ndims[dim], nnz, old_row_ptrs.data(), old_col_ids.data(), values.data(), kernelType,
//                     orderingType, corseningType, false, k, supRowSizes.data());

    CSRk_Graph A_mat(transpose_bitfield.width(), transpose_bitfield.width(), transpose_nnz,
                     row_ptrs_full.data(), col_ids_full.data(), values.data(), kernelType,
                     orderingType, corseningType, false, k, supRowSizes.data());

    A_mat.putInCSRkFormat();



//    //All cols should be in the order they now should appear in, we just need to extract the entire dim's permutation from it.
//    std::vector<std::uint32_t> cprm = find_column_permutation(std::span(rowPtrs + 1, row_size - 1), std::span(colIds, nnz + 1),
//                            std::span(A_mat.get_r_vec(), row_size - 1), std::span(A_mat.get_c_vec(), nnz + 1),
//                            std::span(A_mat.getPermutation(), row_size), ndims[dim]+1);

//All cols should be in the order they now should appear in, we just need to extract the entire dim's permutation from it.
//TODO not needed right now, Only symmetric square matrices are valid ATM.
//    std::vector<std::uint32_t> cprm = find_column_permutation(std::span(old_row_ptrs), std::span(old_col_ids),
//                                                              std::span(A_mat.get_r_vec(), row_size - 1), std::span(A_mat.get_c_vec(), nnz),
//                                                              std::span(A_mat.getPermutation(), row_size - 1), ndims[dim]);
    auto row_perm_span = std::span(A_mat.getPermutation(), row_size - 1);
    std::vector<std::uint32_t> cprm(row_perm_span.begin(), row_perm_span.end());




    //need to move it *back* into being 1 based.
    for(auto& value : cprm){
        value += 1;
    }
    //suppposed to be 1 larger.
    cprm.insert(cprm.begin(), 0);





  //  std::span<unsigned int> original_permutation(A_mat.getPermutation(), row_size);

    //we need to take rows, organize columns?
    // TODO actually should be able to just take cols from A_mat, compress latterally, to figure out order?
    // it *looks* like LexOrderThem is just keeping relative order of empty cols  inbetween, so we should do this as well.

    //TODO get original permutation (should be inside of A_Mat)
    //TODO that permutation becomes what we change for org ids etc...?
    //TODO permBigG is hypothetically ether row or column permutation?
    //TODO totally row permutation because allocates in number of rows.

    // take rows and re-permute with columns?

    //Need to take where items came from, set new locations to current ones (shown in loop in orderit I think)
    //ords is the origional order first, then subsequent orderings later on.

    //TODO, what the heck is CPRM?
//    t0 = u_seconds();
//    lexOrderThem(mtxNrows, ndims[dim], rowPtrs, colIds, cprm);
//    t1 =u_seconds()-t0;
    printf("dim %u lexorder time %.2f\n", dim, t1);
    // printf("cprm: \n");
    // ptiDumpIndexArray(cprm, ndims[dim] + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < ndims[dim]; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[dim][c];
    }
    for (c=0; c < ndims[dim]; c++)
        orgIds[dim][c] = saveOrgIds[cprm[c+1]-1];

    // printf("invcprm: \n");
    // ptiDumpIndexArray(invcprm, ndims[dim] + 1, stdout);

    /*rename the dim component of nonzeros*/
    for (z = 0; z < nnz; z++)
        coords[z][dim] = invcprm[coords[z][dim]];

    free(mode_order);
    free(saveOrgIds);
    free(invcprm);
//    free(cprm);
    free(colIds);
    free(rowPtrs);
}


void orderitBandK(ptiSparseTensor * tsr, ptiIndex ** newIndices, int const renumber, ptiIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.

     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    ptiIndex i, m, nm = tsr->nmodes;
    ptiNnzIndex z, nnz = tsr->nnz;
    ptiIndex ** coords;
    ptiIndex its;

    /* copy the indices */
    ptiTimer copy_coord_timer;
    ptiNewTimer(&copy_coord_timer, 0);
    ptiStartTimer(copy_coord_timer);

    coords = (ptiIndex **) malloc(sizeof(ptiIndex*) * nnz);
    for (z = 0; z < nnz; z++)
    {
        coords[z] = (ptiIndex *) malloc(sizeof(ptiIndex) * nm);
        for (m = 0; m < nm; m++) {
            coords[z][m] = tsr->inds[m].data[z];
        }
    }

    ptiStopTimer(copy_coord_timer);
    ptiPrintElapsedTime(copy_coord_timer, "Copy coordinate time");
    ptiFreeTimer(copy_coord_timer);

    /* checkEmptySlices(coords, nnz, nm, tsr->ndims); */

    if (renumber == 1) {    /* Lexi-order renumbering */

        ptiIndex ** orgIds = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);

        for (m = 0; m < nm; m++)
        {
            orgIds[m] = (ptiIndex *) malloc(sizeof(ptiIndex) * tsr->ndims[m]);
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("old.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nm; m++)
                orderBandK2(coords, nnz, nm, tsr->ndims, m, orgIds);

            // fprintf(stdout, "\niter %u:\n", its);
            // for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            //     ptiDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nm; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        for (m = 0; m < nm; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
        /*
         REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
         but on a few cases it did not help much. Just leaving it in case we want to use it.
         */
        printf("[BFS-like]\n");
        orderforHiCOObfsLike(nm, nnz, tsr->ndims, coords, newIndices);
    }

    // printf("set the new indices\n");
/*    checkNewIndices(newIndices, nm, tsr->ndims);*/

    for (z = 0; z < nnz; z++)
        free(coords[z]);
    free(coords);

}


struct NewIndicesView{
    ptiIndex ** newIndices = nullptr;
};





void orderDimTranslated(std::span<std::vector<ptiIndex>> coords, ptiNnzIndex const nnz, ptiIndex const nm, std::span<ptiIndex> mode_sizes, ptiIndex const dim, std::span<std::vector<ptiIndex>> orgIds);
void orderitTranslated(ptiSparseTensor * tsr, ptiIndex ** newIndices, int const renumber, ptiIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.

     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */

    ptiIndex mode_count = tsr->nmodes;
    ptiNnzIndex nonzero_count = tsr->nnz;

    std::span<ptiIndex> dim_sizes(tsr->ndims, mode_count);

    std::vector<std::vector<ptiIndex>> coords(nonzero_count);
//    ptiIndex ** coords;

    /* copy the indices */
    ptiTimer copy_coord_timer;
    ptiNewTimer(&copy_coord_timer, 0);
    ptiStartTimer(copy_coord_timer);

    for (ptiNnzIndex nz_idx = 0; nz_idx < nonzero_count; nz_idx++)
    {
        coords[nz_idx].resize(mode_count);
        for (ptiIndex mode = 0; mode < mode_count; mode++) {
            coords[nz_idx][mode] = tsr->inds[mode].data[nz_idx];
        }
    }

    ptiStopTimer(copy_coord_timer);
    ptiPrintElapsedTime(copy_coord_timer, "Copy coordinate time");
    ptiFreeTimer(copy_coord_timer);

    /* checkEmptySlices(coords, nnz, nm, tsr->ndims); */

    if (renumber == 1) {    /* Lexi-order renumbering */

        std::vector<std::vector<ptiIndex>> original_idxs(mode_count);
        for (ptiIndex mode = 0; mode < mode_count; mode++) {
            original_idxs[mode].resize(dim_sizes[mode]);
            for (ptiIndex dim_idx = 0; dim_idx < dim_sizes[mode]; dim_idx++) {
                original_idxs[mode][dim_idx] = dim_idx;
            }
        }

        // FILE * debug_fp = fopen("old.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (ptiIndex iteration = 0; iteration < iterations; iteration++){
            printf("[Lexi-order] Optimizing the numbering for its %u\n", iteration+1);
            for (ptiIndex mode = 0; mode < mode_count; mode++) {
                orderDimTranslated(coords, nonzero_count, mode_count, dim_sizes, mode, original_idxs);
            }

            // fprintf(stdout, "\niter %u:\n", its);
            // for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            //     ptiDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (ptiIndex mode = 0; mode < mode_count; mode++) {
            for (ptiIndex dim_idx = 0; dim_idx < dim_sizes[mode]; dim_idx++) {
                newIndices[mode][original_idxs[mode][dim_idx]] = dim_idx;
            }
        }

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
        /*
         REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
         but on a few cases it did not help much. Just leaving it in case we want to use it.
         */
        printf("[BFS-like]\n");

        //orderforHiCOObfsLike(mode_count, nonzero_count, tsr->ndims, coords, newIndices);
    }

    // printf("set the new indices\n");
/*    checkNewIndices(newIndices, nm, tsr->ndims);*/

}


/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

// SB:
// coords is the SOA x,y,z... list of COO indexes
// nnz = number of non zeros
// nm = number of modes
// ndims = the length of each mode/dimension (length of x, y, and z)
// dim = the given dimension processing.
// orgIds = the origional order of all the IDs (same size as coords)
void orderDimTranslated(std::span<std::vector<ptiIndex>> coords, ptiNnzIndex const nonzero_count, ptiIndex const mode_count,
                        std::span<ptiIndex> mode_sizes, ptiIndex const mode_selected, std::span<std::vector<ptiIndex>> orgIds){

    std::vector<ptiIndex> mode_order((mode_count - 1));
    ptiIndex i = 0;
    for(ptiIndex mode = 0; mode < mode_count; ++mode) {
        if (mode != mode_selected) {
            mode_order[i] = mode;
            ++ i;
        }
    }

    double t1, t0;
    t0 = u_seconds();
    // mySort(coords,  nnz-1, nm, ndims, dim);
    //SB:
    // sorts coords according to all dims except dim, where items are refered with newIndices
    // an iterative quicksort
    // appears to only be relevant to creating CSR version
    //TODO deal with this?
  //  mySortFast(coords,  nonzero_count-1, mode_count, mode_sizes, mode_selected, mode_order);
    t1 = u_seconds()-t0;
    printf("dim %u, sort time %.2f\n", mode_selected, t1);
    // printCoords(coords, nnz, nm);
    /* we matricize this (others x thisDim), whose columns will be renumbered */

    /* on the matrix all arrays are from 1, and all indices are from 1. */

    std::vector<ptiNnzIndex> rowPtrs(nonzero_count+2); /*large space*/
    std::vector<ptiIndex> colIds ((nonzero_count+2)); /*large space*/

    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs[1] = 1;

    colIds[1] = coords[0][mode_selected] + 1;
    ptiNnzIndex atRowPlus1 = 2;
    ptiNnzIndex mtrxNnz = 2;/* start filling from the second element */

    t0 = u_seconds();
    //SB:
    //generates row pointers for CSR and increments col ids.
    //lexigraphically compares the indexes for the coordinates first.
    for (ptiNnzIndex nz_idx = 1; nz_idx < nonzero_count; nz_idx++)
    {
        // if(isLessThanOrEqualTo( coords[z], coords[z-1], nm, ndims, dim) != 0)
        if(isLessThanOrEqualToFast( coords[nz_idx].data(), coords[nz_idx-1].data(), mode_count, mode_order.data()) != 0) {
            rowPtrs[atRowPlus1++] = mtrxNnz; /* close the previous row and start a new one. */
        }
        colIds[mtrxNnz++] = coords[nz_idx][mode_selected]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    ptiNnzIndex mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("dim %u create time %.2f\n", mode_selected, t1);

    rowPtrs.resize((mtxNrows+2));
#ifdef TEST_CSR_ORDER_OUTPUT
    {
        std::cout << "row pointers:" << std::endl;
        for (std::size_t idx = 0; idx < (mtxNrows + 2); ++idx) {
            std::cout << rowPtrs[idx] << ", ";
        }
        std::cout << std::endl;
        std::cout << "nnz " << nonzero_count << " mtxNrows " << mtxNrows << std::endl;
//        if ((mtxNrows + 2) == (nnz + 2)) {
//            std::cout << "nnz and mtxNrows match" << std::endl;
//        }
        std::cout << "colIds:" << std::endl;
        for (std::size_t idx = 0; idx < (nonzero_count + 2); ++idx) {
            std::cout << colIds[idx] << ", ";
        }
        std::cout << std::endl;
    }
#endif
    std::vector<ptiIndex> cprm(mode_sizes[mode_selected] + 1);
    std::vector<ptiIndex> invcprm(mode_sizes[mode_selected] + 1);
    std::vector<ptiIndex> saveOrgIds(mode_sizes[mode_selected] + 1);
    /*    checkRepeatIndex(mtxNrows, rowPtrs, colIds, ndims[dim] );*/

    // printf("rowPtrs: \n");
    // ptiDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // ptiDumpIndexArray(colIds, nnz + 2, stdout);

    t0 = u_seconds();
    //TODO is cprm, column permutation?
    lexOrderThem(mtxNrows, mode_sizes[mode_selected], rowPtrs.data(), colIds.data(), cprm.data());
    t1 =u_seconds()-t0;
    printf("dim %u lexorder time %.2f\n", mode_selected, t1);
    // printf("cprm: \n");
    // ptiDumpIndexArray(cprm, ndims[dim] + 1, stdout);

    /* update orgIds and modify coords */

    for (ptiIndex mode_length_idx =0; mode_length_idx < mode_sizes[mode_selected]; mode_length_idx++){
        //first figure out inverse mapping to column permutation.
        invcprm[cprm[mode_length_idx+1]-1] = mode_length_idx;
        //in order assignment for given mode "column"?
        saveOrgIds[mode_length_idx] = orgIds[mode_selected][mode_length_idx];
    }

    for (ptiIndex mode_length_idx=0; mode_length_idx < mode_sizes[mode_selected]; mode_length_idx++){
        //using column permutation keep track of orgIds (how to map back to the original ids?)
        orgIds[mode_selected][mode_length_idx] = saveOrgIds[cprm[mode_length_idx+1]-1];
    }

    // printf("invcprm: \n");
    // ptiDumpIndexArray(invcprm, ndims[dim] + 1, stdout);

    /*rename the dim component of nonzeros*/
    for (ptiNnzIndex nz_idx = 0; nz_idx < nonzero_count; nz_idx++) {
        //update coords with inverse column permutation.
        coords[nz_idx][mode_selected] = invcprm[coords[nz_idx][mode_selected]];
    }
}