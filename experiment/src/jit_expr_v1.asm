  pushq   %rbp
  unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
  movq    %rsp, %rbp
  unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
  subq    %rsp, $48, %rsp
block0:
  movdqu  %xmm2, rsp(8 + virtual offset)
  movdqu  %xmm3, rsp(24 + virtual offset)
  load_ext_name userextname0+0, %rcx
  call    *%rcx
  load_ext_name userextname1+0, %rcx
  movdqu  rsp(8 + virtual offset), %xmm1
  call    *%rcx
  lea     rsp(0 + virtual offset), %rdi
  vmovsd  %xmm0, 0(%rdi)
  lea     rsp(1 + virtual offset), %rsi
  movdqu  rsp(24 + virtual offset), %xmm3
  vmovsd  %xmm3, 0(%rsi)
  load_ext_name userextname2+0, %r8
  call    *%r8
  addq    %rsp, $48, %rsp
  movq    %rbp, %rsp
  popq    %rbp
  ret