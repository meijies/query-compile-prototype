  pushq   %rbp
  unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
  movq    %rsp, %rbp
  unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
block0:
  vaddsd  %xmm0, %xmm1, %xmm7
  vdivsd  %xmm7, %xmm2, %xmm7
  ucomisd %xmm7, %xmm3
  setnbe  %al
  movq    %rbp, %rsp
  popq    %rbp
  ret