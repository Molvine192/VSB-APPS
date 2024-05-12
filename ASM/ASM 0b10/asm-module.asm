bits 64

    section .data
    extern count

    section .text

;zadani 1
    global my_strchr
my_strchr:
    enter 0,0
    mov rax, -1
    mov rcx, 0
    mov r8, 0

    cmp sil, 'A'
    jl .for
    cmp sil, 'Z'
    jg .for
    add sil, 32  
.for:
    mov dl, [rdi + rcx]
    test dl, dl
    je .end

    cmp dl, 'A'
    jl .notuprc
    cmp dl, 'Z'
    jg .notuprc
    add dl, 32
.notuprc:
    cmp dl, sil
    je .match
    jne .next
.match:
    inc r8
    cmp rax, -1
    jg .next
    mov rax, rcx
.next:
    inc rcx
    jmp .for
.end:
    mov [count], r8
    leave
    ret

; zadani 2
    global str2int
str2int:
	enter 0,0
	mov rax, 0 
	mov rdx, 0
	mov r9, 0	
	mov r10, 0
	mov [rsi], byte 0
.for:
	cmp [rdi + rdx * 1 ], byte 0
	je .end
	cmp rax, 0
	jne .shift
	sub [rdi + rdx * 1], byte '0'
	movsx r10, byte [rdi + rdx * 1]
	add rax, r10
	inc rdx
	jmp .for
.shift:
	mov r10, 0
	mov r9, rax
	shl rax, 3
	shl r9, 1
	add rax, r9
	movsx r10, byte [rdi + rdx * 1]
	sub r10, byte '0'
	add rax, r10
	inc rdx
	jmp .for
.end:
	mov [rsi], rax
	leave
	ret
    

; zadani 3
    global not_bits
not_bits:
    enter 0,0
    
    mov r9, [rdi]   
    mov r8, 1      
    mov rcx, 0      
    mov r10, 0     
.for:
    cmp rcx, 64
    jge .end
    test r9, r8
    je .no
    inc r10
.no:
    mov r11, 0  
.for_2:
    cmp r11, rdx
    jge .end_2
    mov rax, 0
    mov al, [rsi + r11 * 1]
    cmp rcx, rax
    je .dn
    inc r11
    jmp .for_2
.end_2:
    jmp .daal
.dn:
    xor r9, r8
    jmp .daal
.daal:
    inc rcx
    shl r8, 1
    jmp .for
.end:
    mov [rdi], r9
  
    leave
    ret