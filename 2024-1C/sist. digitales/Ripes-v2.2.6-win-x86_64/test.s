main:
addi x11 x0 4
lw x12 0 x11
addi x13 x0 4
lw x13 0 x13
lw x13 0 x13
beq x13 x12 -20
guardar:
lui x14 0xfffa6
addi x14 x14 -1539
add x12 x14 x12
sw x11 40 x12
fin_programa:
addi x10 x0 0
addi x17 x0 93
ecall