/* Ordered, encoded frame sizes; synthetic payload. Full readback outside timing. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/uio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
typedef struct{uint32_t d[8],item,div;uint64_t offset,packed;}Header;
typedef struct{Header h;size_t bytes;}Frame;
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void die(const char*s){perror(s);exit(1);}
static void vectors(int fd,struct iovec*v,int count){while(count){ssize_t n=writev(fd,v,count);if(n<0&&errno==EINTR)continue;if(n<=0)die("writev");while(count&&(size_t)n>=v->iov_len){n-=v->iov_len;v++;count--;}if(n){v->iov_base=(char*)v->iov_base+n;v->iov_len-=n;}}}
int main(int argc,char**argv){if(argc!=4)return 1;FILE*in=fopen(argv[1],"r");if(!in)die("input");Frame*fs=calloc(200000,sizeof(Frame));size_t count=0,total=8,max=0;
while(count<200000){Frame*f=fs+count;unsigned long long packed;size_t bytes;int n=fscanf(in,"%u %u %u %u %u %u %u %zu %llu",f->h.d,f->h.d+1,f->h.d+2,f->h.d+3,f->h.d+4,f->h.d+5,f->h.d+6,&bytes,&packed);if(n==EOF)break;if(n!=9)die("layout");f->h.d[7]=1;f->h.item=16;f->h.div=32;f->h.offset=48;f->h.packed=packed;f->bytes=bytes;total+=56+bytes;if(bytes>max)max=bytes;count++;}fclose(in);if(!count||sizeof(Header)!=56)return 2;
char*source=malloc(max);if(!source)die("source");for(size_t i=0;i<max;i++)source[i]=(char)(i*131+17);
const char*names[]={"stdio64k","stdio256k","stdio1m","stdio4m","writev64frames","mmap_cold","mmap_pretouch","anonymous_pretouch"};size_t sizes[]={65536,262144,1048576,4194304};int comma=0;
printf("{\"synthetic_payload\":true,\"ordered_frames\":%zu,\"bytes\":%zu,\"results\":[",count,total);
for(int r=0;r<atoi(argv[3]);r++)for(int k=0;k<8;k++){int mode=(k+r)%8;char path[1024];snprintf(path,sizeof(path),"%s/micro-%d-%d.bin",argv[2],r,mode);FILE*fp=NULL;int fd=-1;char*dest=NULL,*buffer=NULL;
uint64_t setup=now();if(mode<4){buffer=malloc(sizes[mode]);if(!buffer)die("buffer");}
if(mode>=5){if(mode!=7){fd=open(path,O_CREAT|O_EXCL|O_RDWR,0600);if(fd<0||ftruncate(fd,total))die("file mapping");}dest=mmap(NULL,total,PROT_READ|PROT_WRITE,mode==7?MAP_PRIVATE|MAP_ANONYMOUS:MAP_SHARED,fd,0);if(dest==MAP_FAILED)die("mmap");if(mode>=6)memset(dest,0,total);}setup=now()-setup;
struct rusage a,b;getrusage(RUSAGE_SELF,&a);uint64_t wall=now();size_t pos=8,syscalls=0;
if(mode<4){fp=fopen(path,"wb");if(!fp||setvbuf(fp,buffer,_IOFBF,sizes[mode]))die("stdio");if(fwrite("NTGPU13\n",1,8,fp)!=8)die("magic");}
else if(mode==4){fd=open(path,O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0||write(fd,"NTGPU13\n",8)!=8)die("open");syscalls++;}
else memcpy(dest,"NTGPU13\n",8);
for(size_t i=0;i<count;){if(mode==4){struct iovec v[128];int n=0;for(int j=0;j<64&&i<count;j++,i++){v[n++]=(struct iovec){&fs[i].h,56};v[n++]=(struct iovec){source,fs[i].bytes};pos+=56+fs[i].bytes;}vectors(fd,v,n);syscalls++;}
else{Frame*f=fs+i;if(mode<4){if(fwrite(&f->h,1,56,fp)!=56||fwrite(source,1,f->bytes,fp)!=f->bytes)die("fwrite");}else{memcpy(dest+pos,&f->h,56);memcpy(dest+pos+56,source,f->bytes);}pos+=56+f->bytes;i++;}}
if(fp&&fclose(fp))die("close");if(fd>=0&&close(fd))die("close");wall=now()-wall;getrusage(RUSAGE_SELF,&b);if(pos!=total)return 3;
// Read the complete receiver and compare all bytes, not just final length.
char*view=dest;int checkfd=-1;if(mode<5){checkfd=open(path,O_RDONLY);if(checkfd<0)die("check open");struct stat st;if(fstat(checkfd,&st)||st.st_size!=total)return 4;view=mmap(NULL,total,PROT_READ,MAP_SHARED,checkfd,0);if(view==MAP_FAILED)die("check map");}
if(memcmp(view,"NTGPU13\n",8))return 5;pos=8;for(size_t i=0;i<count;i++){if(memcmp(view+pos,&fs[i].h,56)||memcmp(view+pos+56,source,fs[i].bytes))return 6;pos+=56+fs[i].bytes;}
if(munmap(view,total))die("unmap");if(checkfd>=0)close(checkfd);if(mode!=7&&unlink(path))die("unlink");free(buffer);
printf("%s{\"round\":%d,\"mode\":\"%s\",\"setup_ns\":%llu,\"wall_ns\":%llu,\"minor_faults\":%ld,\"major_faults\":%ld,\"user_s\":%.9f,\"system_s\":%.9f,\"verified_bytes\":%zu,\"explicit_writev_groups\":%zu}",comma?",":"",r+1,names[mode],(unsigned long long)setup,(unsigned long long)wall,b.ru_minflt-a.ru_minflt,b.ru_majflt-a.ru_majflt,(b.ru_utime.tv_sec-a.ru_utime.tv_sec)+(b.ru_utime.tv_usec-a.ru_utime.tv_usec)/1e6,(b.ru_stime.tv_sec-a.ru_stime.tv_sec)+(b.ru_stime.tv_usec-a.ru_stime.tv_usec)/1e6,total,syscalls);comma=1;fflush(stdout);
}puts("],\"full_content_verified\":true}");free(fs);free(source);return 0;}
