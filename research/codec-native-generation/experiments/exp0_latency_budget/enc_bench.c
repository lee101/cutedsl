#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef size_t (*enc_t)(const unsigned char*, int, int, int, float, unsigned char**);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(int argc,char**argv){
  int W=argc>1?atoi(argv[1]):512, H=W, N=20;
  void*l=dlopen("libwebp.so.7",RTLD_NOW); enc_t e=(enc_t)dlsym(l,"WebPEncodeRGB");
  unsigned char*buf=malloc((size_t)W*H*3);
  for(size_t i=0;i<(size_t)W*H*3;i++) buf[i]=(unsigned char)((i*7919)%251);
  double best=1e9,tot=0;
  size_t sz=0;
  for(int i=0;i<N;i++){unsigned char*o=NULL;double s=ms();sz=e(buf,W,H,W*3,85.0f,&o);double d=ms()-s;tot+=d;if(d<best)best=d;free(o);}
  printf("webp q85 %dx%d: mean=%.1fms best=%.1fms out=%zu bytes\n",W,W,tot/N,best,sz);
  free(buf);return 0;
}
