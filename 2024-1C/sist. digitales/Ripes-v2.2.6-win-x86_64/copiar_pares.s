.data:
    q: .word 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08
    s: .word 0x0a 0x0b 0x0c 0x0d 0x0f 0xff 0xff 0xff
    
.text:
    la a0 q
    la a1 s
    addi a2 zero 8
    comienzo:
        addi a3 zero 1 #declaramos un nro con 1 bit
        beq a2 zero 52 #si ya recorrimos todo el arr, terminamos
        addi a2 a2 -1 #caso contrario, restamos 1 a length y resolvemos
        lw t0 0(a0) #cargamos el primer elemento de a0
        slli t0 t0 31 #le "borramos" 31 bits, si el 32 es 1 entonces es par, else impar
        beq t0 a3 20 # t0 == 1?
        sw x0 0(a0) #si es impar, guardamos 0 
        addi a0 a0 4
        addi a1 a1 4
        beq zero zero -32 #volvemos a loopear
        sw t0 0(a0) #si es par, guardamos t0
        addi a0 a0 4
        addi a1 a1 4
        beq zero zero -44 #volvemos a loopear
        
        
    fin:
        add a0 a1 zero
        