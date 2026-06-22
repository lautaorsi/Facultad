.data:
    q: .word 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09
    
.text:
    la a0 q
    addi a1 zero 9
    addi t0 zero 0
    comienzo:
        beq a1 zero fin
        lw t0 0(a0)
        addi a0 a0 4
        addi a1 a1 -1
        add t1 t1 t0
        j comienzo
    fin:
        add a0 zero t1