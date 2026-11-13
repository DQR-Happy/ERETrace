/* Independent output experiments on the original T8 ring and scheduling path. */
#ifndef NEUTRINO_LEGACY_CAPTURE_H
#define NEUTRINO_LEGACY_CAPTURE_H
typedef struct {
    bulk_engine output;CUmodule module;CUfunction encoder;CUdeviceptr error,scratch;
    uint64_t encoder_calls,encoder_bytes,verified_frames,packed_capacity,verify_capacity;
} legacy_state;
static legacy_state *legacy_get(async_engine*e){
    if(e->output_state)return e->output_state;
    legacy_state*s=calloc(1,sizeof(*s));if(!s)async_fatal("legacy output metadata");
    s->output.directory=strdup(e->directory);if(!s->output.directory)async_fatal("legacy output directory");
    if(legacy_output_mode==3){
        const char*path=getenv("NEUTRINO_CAPTURE_ENCODER");if(!path)async_fatal("encoder cubin path missing");
        CUDA_CHECK(real_cuModuleLoad(&s->module,path));CUDA_CHECK(real_cuModuleGetFunction(&s->encoder,s->module,"neutrino_capture_pack"));
        CUDA_CHECK(real_cuMemAlloc_v2(&s->error,4));CUDA_CHECK(real_cuMemsetD32_v2(s->error,0,1));
    }
    e->output_state=s;return s;
}
static void legacy_copy(async_engine*e,async_slot*h,CUstream stream){
    h->codec=0;h->copy_bytes=h->bytes;
    if(legacy_output_mode){
        cd_tick prepare_tick=cd_begin();
        legacy_state*s=legacy_get(e);
        if(legacy_output_mode==3 && h->bytes>=65536){
            size_t n=h->bytes/16,packed=n*13;
            if(s->packed_capacity<packed){
                if(s->scratch){CUDA_CHECK(real_cuStreamSynchronize(stream));CUDA_CHECK(real_cuMemFree_v2(s->scratch));}
                CUDA_CHECK(real_cuMemAlloc_v2(&s->scratch,packed));s->packed_capacity=packed;
            }
            h->encoded=s->scratch;
            unsigned long long count=n;void*args[]={&h->device,&s->scratch,&count,&s->error};
            cd_record(CD_ENCODER_PREP,prepare_tick);prepare_tick.wall=0;
            if(!cd_no_pack){CD_RUN(ENCODER,CUDA_CHECK(real_cuLaunchKernel(s->encoder,(unsigned)((n+255)/256),1,1,256,1,1,0,stream,args,NULL)));s->encoder_calls++;s->encoder_bytes+=packed;}
            h->codec=1;h->copy_bytes=packed;
            if(legacy_verify_encoder && h->verify_capacity<h->bytes){
                if(h->verify_host)CUDA_CHECK(real_cuMemFreeHost(h->verify_host));
                CUDA_CHECK(real_cuMemAllocHost_v2(&h->verify_host,h->bytes));s->verify_capacity+=h->bytes-h->verify_capacity;h->verify_capacity=h->bytes;
            }
        }
        cd_record(CD_ENCODER_PREP,prepare_tick);
    }
    if(!cd_no_d2h){CD_RUN(COPY,CUDA_CHECK(real_cuMemcpyDtoHAsync_v2(h->host,h->codec?h->encoded:h->device,h->copy_bytes,stream)));}
    if(h->codec && legacy_verify_encoder)CUDA_CHECK(real_cuMemcpyDtoHAsync_v2(h->verify_host,h->device,h->bytes,stream));
}
static void legacy_write(async_engine*e,async_slot*h){
    legacy_state*s=e->output_state;bulk_engine*out=&s->output;
    if(cd_consume==2){out->calls++;out->payload+=h->bytes;out->logical_trace+=h->bytes+48;return;}
    bulk_frame f;memcpy(f.dims,h->dims,7*sizeof(unsigned));f.offset=0;f.bytes=h->bytes;
    if(legacy_output_mode!=3){bulk_host view;view.host=h->host;bulk_output_frame(out,&view,&f);return;}
    if(h->codec && legacy_verify_encoder){
        void*p=realloc(out->decoded,h->bytes);if(!p)async_fatal("GPU codec verify buffer");out->decoded=p;
        size_t n=h->bytes/16;char*raw=h->host;
        for(size_t i=0;i<n;i++){uint64_t start;uint32_t elapsed,sm=(unsigned char)raw[n*12+i];
            memcpy(&start,raw+i*8,8);memcpy(&elapsed,raw+n*8+i*4,4);
            memcpy((char*)p+i*16,&start,8);memcpy((char*)p+i*16+8,&elapsed,4);memcpy((char*)p+i*16+12,&sm,4);}
        if(memcmp(p,h->verify_host,h->bytes))async_fatal("GPU lossless encoding mismatch");s->verified_frames++;
    }
    uint64_t tick=ab_stats?async_clock(CLOCK_MONOTONIC):0;
    uint64_t storage_before=out->storage_bytes;
    if(cd_consume==1){
        if(!out->ram){out->ram_capacity=768ULL*1024*1024;out->ram=malloc(out->ram_capacity);if(!out->ram)async_fatal("encoded RAM allocation");}
        if(!out->ram_used){CD_RUN(OPEN,bulk_put(out,"NTGPU13\n",8));}
    }else if(!out->fp){CD_RUN(OPEN,{char*path=malloc(strlen(out->directory)+64);sprintf(path,"%s/legacy-%llu.nts",out->directory,(unsigned long long)out->file_seq++);
        out->fp=fopen(path,"wb");free(path);out->stdio_buffer=malloc(1048576);
        if(!out->fp||!out->stdio_buffer||setvbuf(out->fp,out->stdio_buffer,_IOFBF,1048576))async_fatal("GPU encoded output open");bulk_put(out,"NTGPU13\n",8);});}
    trace_header_t th={h->dims[0],h->dims[1],h->dims[2],h->dims[3],h->dims[4],h->dims[5],h->dims[6],1};trace_section_t ts={16,32,48};
    uint64_t packed=h->codec?h->copy_bytes:0;
    CD_RUN(HEADER,{bulk_put(out,&th,sizeof(th));bulk_put(out,&ts,sizeof(ts));bulk_put(out,&packed,8);});
    CD_RUN(PAYLOAD,bulk_put(out,h->host,h->copy_bytes));
    if(cd_active())e->detail->groups[h->detail_group].storage_bytes+=out->storage_bytes-storage_before;
    out->calls++;out->payload+=h->bytes;out->logical_trace+=h->bytes+48;if(ab_stats)out->write_ns+=async_clock(CLOCK_MONOTONIC)-tick;
}
static int legacy_boundary(async_engine*e){
    legacy_state*s=e->output_state;if(!s)return 0;bulk_engine*out=&s->output;int bad=0;
    if(out->fp){CD_RUN(CLOSE,{bad=ferror(out->fp);if(fclose(out->fp))bad=1;out->fp=NULL;free(out->stdio_buffer);out->stdio_buffer=NULL;});}
    if(bulk_sink && out->ram_used){out->ram_last_size=out->ram_used;out->ram_used=0;}
    if(legacy_output_mode==3){unsigned error=0;CD_RUN(ERROR_COPY,CUDA_CHECK(real_cuMemcpyDtoH_v2(&error,s->error,4)));if(error)async_fatal("SM ID does not fit lossless GPU encoding");}
    return bad?-1:0;
}
static void legacy_reset(async_engine*e){
    legacy_state*s=e->output_state;if(!s)return;bulk_engine*out=&s->output;
    out->calls=out->payload=out->logical_trace=out->storage_bytes=out->write_ns=out->encode_ns=out->verify_ns=0;
    out->codec_verified_frames=out->codec_raw_frames=out->codec_zero_records=out->codec_global_wide_frames=0;
    s->encoder_calls=s->encoder_bytes=s->verified_frames=0;
}
static void legacy_json(FILE*fp){
    fputs(",\"legacy_outputs\":[",fp);int comma=0;
    for(async_engine*e=async_engines;e;e=e->next){legacy_state*s=e->output_state;if(!s)continue;bulk_engine*out=&s->output;
        fprintf(fp,"%s{\"frames\":%llu,\"payload_bytes\":%llu,\"logical_trace_bytes\":%llu,\"storage_bytes\":%llu,\"worker_write_ns\":%llu,\"worker_encode_ns\":%llu,\"codec_verify_ns\":%llu,\"encoder_calls\":%llu,\"encoder_bytes\":%llu,\"gpu_verified_frames\":%llu,\"packed_device_capacity_bytes\":%llu,\"validation_host_capacity_bytes\":%llu}",comma?",":"",
            (unsigned long long)out->calls,(unsigned long long)out->payload,(unsigned long long)out->logical_trace,(unsigned long long)out->storage_bytes,
            (unsigned long long)out->write_ns,(unsigned long long)out->encode_ns,(unsigned long long)out->verify_ns,
            (unsigned long long)s->encoder_calls,(unsigned long long)s->encoder_bytes,(unsigned long long)s->verified_frames,(unsigned long long)s->packed_capacity,(unsigned long long)s->verify_capacity);comma=1;
    }fputc(']',fp);
}
static void legacy_release(async_engine*e){
    legacy_state*s=e->output_state;if(!s)return;bulk_engine*out=&s->output;
    if(bulk_sink && out->ram_last_size){char*path=malloc(strlen(out->directory)+32);sprintf(path,"%s/ram-last.%s",out->directory,legacy_output_mode==3?"nts":"bin");FILE*fp=fopen(path,"wb");free(path);
        if(!fp||fwrite(out->ram,1,out->ram_last_size,fp)!=out->ram_last_size||fclose(fp))async_fatal("legacy RAM validation dump");}
    if(s->module){if(s->scratch)CUDA_CHECK(real_cuMemFree_v2(s->scratch));CUDA_CHECK(real_cuMemFree_v2(s->error));CUDA_CHECK(real_cuModuleUnload(s->module));}
    free(out->encoded);free(out->decoded);free(out->ram);free(out->directory);free(s);e->output_state=NULL;
}
#endif
