  pushq   %rbp
  unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
  movq    %rsp, %rbp
  unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
  subq    %rsp, $16, %rsp
block0:
  load_ext_name userextname0+0, %r8
  call    *%r8
  movabsq $4613937818241073152, %rdx
  vmovq   %rdx, %xmm1
  load_ext_name userextname1+0, %r8
  call    *%r8
  lea     rsp(0 + virtual offset), %rdi
  vmovsd  %xmm0, 0(%rdi)
  movabsq $4616189618054758400, %r8
  vmovq   %r8, %xmm0
  lea     rsp(1 + virtual offset), %rsi
  vmovsd  %xmm0, 0(%rsi)
  load_ext_name userextname2+0, %r10
  call    *%r10
  addq    %rsp, $16, %rsp
  movq    %rbp, %rsp
  popq    %rbp
  ret