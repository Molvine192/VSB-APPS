bits 64

    section .data

    section .text

;zadani 1
    global fill_pyramid_numbers
fill_pyramid_numbers:
    enter 0,0
    push rbx
    movsx rsi, esi
    mov rcx, 0
.back:
    cmp rcx, rsi
    jge .end
    mov r8, 0
    mov rbx, 0
    mov r10, rcx
    inc r10
.inner_back:
    cmp rbx, r10
    jge .inner_end
    inc rbx
    mov rax, rbx
    mul rbx
    add r8, rax
    jmp .inner_back
.inner_end:
    mov [rdi + rcx * 8], r8
    inc rcx
    jmp .back
.end:
    pop rbx
    leave
    ret

; zadani 2
    global multiples
multiples:
    mov rbx, rdx
    xor r8, r8
    xor rcx, rcx
.loop:
    cmp rcx, rsi
    jge .leave
    mov rax, [rdi + rcx * 8]
    xor rdx, rdx
    div rbx
    test rdx, rdx; 
    jnz .issue
    jmp .next
.issue:
    sub [rdi + rcx * 8], rdx
    inc r8 
.next:
    inc rcx
    jmp .loop
.leave:
    mov rax, r8
    ret

; zadani 4
    global change_array_by_avg
change_array_by_avg:
    enter 0,0
    mov rcx, 0
    mov rax, 0
    movsx rsi, esi
.back:
    cmp rcx, rsi
    jge .end
    add rax, [rdi + rcx * 8]
    inc rcx
    jmp .back
.end:
    cdq
    idiv rsi
    mov rcx, 0
.back2:
    cmp rcx, rsi
    jge .end2
    mov r8, [rdi + rcx * 8]
    cmp r8, rax
    mov r9, -1
    mov r10, 0
    mov r11, 1
    cmovl r8, r9
    cmove r8, r10
    cmovg r8, r11
    mov [rdi + rcx * 8], r8 
    inc rcx
    jmp .back2
.end2:
    leave
    ret