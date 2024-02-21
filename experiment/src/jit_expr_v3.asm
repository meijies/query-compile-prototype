  pushq   %rbp
  unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
  movq    %rsp, %rbp
  unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
block0:
  movq    %rdx, %rax
  vpshufd $68, %xmm0, %xmm3
  vpshufd $68, %xmm1, %xmm2
  jmp     label1
block1:
  vmovupd 0(%rdi), %xmm4
  vaddpd  %xmm4, 0(%rsi), %xmm4
  vdivpd  %xmm4, %xmm3, %xmm4
  vcmppd  $1, %xmm4, %xmm2, %xmm4
  vmovdqu %xmm4, 0(%rdx)
  lea     1(%rcx), %rcx
  lea     16(%rdi), %rdi
  lea     16(%rsi), %rsi
  lea     2(%rdx), %rdx
  cmpq    %r8, %rcx
  jl      label2; j label3
block2:
  jmp     label1
block3:
  movq    %rbp, %rsp
  popq    %rbp
  ret
