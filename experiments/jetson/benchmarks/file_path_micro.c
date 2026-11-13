/* CPU-only output mechanism test, using the measured Qwen frame-size histogram.
 * Same headers/bytes for file and full-size memory receivers. Pre-touch is timed
 * separately and must be added back when discussing full startup cost.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
typedef struct{uint32_t dims[8];uint32_t stride,div;uint64_t offset;} header;
typedef struct{header h;size_t count,bytes;} group;
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
int main(int argc,char**argv){
    if(argc!=4)return 1;FILE*input=fopen(argv[1],"r");if(!input)return 1;
    group groups[512];int ng=0;size_t total=0,frames=0,max=0;
    while(ng<512){group*g=&groups[ng];if(fscanf(input,"%u %u %u %u %u %u %u %zu %zu",&g->h.dims[0],&g->h.dims[1],&g->h.dims[2],&g->h.dims[3],&g->h.dims[4],&g->h.dims[5],&g->h.dims[6],&g->count,&g->bytes)!=9)break;
        g->h.dims[7]=1;g->h.stride=16;g->h.div=32;g->h.offset=48;frames+=g->count;total+=g->count*(g->bytes+48);if(g->bytes>max)max=g->bytes;ng++;}
    fclose(input);if(sizeof(header)!=48 || !ng)return 1;char *source=malloc(max);if(!source)return 1;memset(source,1,max);
    int rounds=atoi(argv[3]);const char*modes[]={"per_frame_file","request_file_flush","request_file_1mib","reused_small_ram_sink","mmap_tmpfs_cold","mmap_tmpfs_pretouched","full_anonymous_pretouched"};
    printf("{\"rounds\":%d,\"frames_per_run\":%zu,\"bytes_per_run\":%zu,\"payload_bytes_per_run\":%zu,\"results\":[",rounds,frames,total,total-48*frames);int comma=0;
    for(int r=0;r<rounds;r++)for(int mi=0;mi<7;mi++){
        int mode=(mi+r)%7;char path[1024];snprintf(path,sizeof(path),"%s/r%d-m%d.bin",argv[2],r,mode);
        FILE*fp=NULL;void*dest=NULL;char*stdio_buffer=NULL;uint64_t setup=now(),touch=0;int fd=-1;
        if(mode==2){stdio_buffer=malloc(1048576);if(!stdio_buffer)return 1;}
        if(mode==3){dest=malloc(max);if(!dest)return 1;memset(dest,0,max);}
        if(mode>=4){
            int flags=MAP_PRIVATE|MAP_ANONYMOUS;
            if(mode!=6){fp=fopen(path,"w+b");if(!fp)return 1;fd=fileno(fp);if(ftruncate(fd,total))return 1;flags=MAP_SHARED;}
            dest=mmap(NULL,total,PROT_READ|PROT_WRITE,flags,fd,0);if(dest==MAP_FAILED)return 1;
            if(mode==5 || mode==6){uint64_t t=now();memset(dest,0,total);touch=now()-t;}
        }
        setup=now()-setup;
        struct rusage a,b;getrusage(RUSAGE_SELF,&a);uint64_t wall=now();size_t position=0,sequence=0,files=0;
        if(mode==1||mode==2){fp=fopen(path,"wb");if(!fp)return 1;files++;if(mode==2 && setvbuf(fp,stdio_buffer,_IOFBF,1048576))return 1;}
        for(int k=0;k<ng;k++){group*g=&groups[k];for(size_t j=0;j<g->count;j++){
            if(mode==0){snprintf(path,sizeof(path),"%s/r%d-m0-%zu.bin",argv[2],r,sequence);fp=fopen(path,"wb");if(!fp)return 1;files++;}
            if(mode<3){if(fwrite(&g->h,48,1,fp)!=1 || fwrite(source,1,g->bytes,fp)!=g->bytes)return 1;if(mode==0){if(fclose(fp))return 1;}else if(mode==1 && fflush(fp))return 1;}
            else if(mode==3){memcpy(dest,source,g->bytes);__asm__ __volatile__(""::"r"(dest):"memory");}
            else{memcpy((char*)dest+position,&g->h,48);memcpy((char*)dest+position+48,source,g->bytes);__asm__ __volatile__(""::"r"(dest):"memory");}
            position+=g->bytes+48;sequence++;
        }}
        if(mode==1||mode==2){if(fclose(fp))return 1;fp=NULL;}
        if(mode>=4){if(munmap(dest,total))return 1;dest=NULL;if(fp){if(fclose(fp))return 1;fp=NULL;files++;}}
        wall=now()-wall;getrusage(RUSAGE_SELF,&b);if(position!=total||sequence!=frames)return 1;
        free(stdio_buffer);
        /* Validate physical output size outside timed window, then delete only our files. */
        size_t physical=0;
        if(mode==0)for(size_t j=0;j<frames;j++){snprintf(path,sizeof(path),"%s/r%d-m0-%zu.bin",argv[2],r,j);struct stat s;if(stat(path,&s))return 1;physical+=s.st_size;if(unlink(path))return 1;}
        else if(mode!=3 && mode!=6){struct stat s;if(stat(path,&s))return 1;physical=s.st_size;if(unlink(path))return 1;}
        if(mode!=3 && mode!=6 && physical!=total)return 1;
        printf("%s{\"round\":%d,\"mode\":\"%s\",\"wall_ns\":%llu,\"setup_ns\":%llu,\"pretouch_ns\":%llu,\"file_count\":%zu,\"physical_file_bytes\":%zu,\"minor_faults\":%ld,\"major_faults\":%ld,\"user_s\":%.9f,\"system_s\":%.9f}",comma?",":"",r+1,modes[mode],(unsigned long long)wall,(unsigned long long)setup,(unsigned long long)touch,files,physical,b.ru_minflt-a.ru_minflt,b.ru_majflt-a.ru_majflt,(b.ru_utime.tv_sec-a.ru_utime.tv_sec)+(b.ru_utime.tv_usec-a.ru_utime.tv_usec)/1e6,(b.ru_stime.tv_sec-a.ru_stime.tv_sec)+(b.ru_stime.tv_usec-a.ru_stime.tv_usec)/1e6);comma=1;if(mode==3)free(dest);
    }puts("],\"size_validation\":true,\"synthetic_payload\":true}");free(source);return 0;
}
