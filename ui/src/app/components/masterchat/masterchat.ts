import { Component, AfterViewChecked, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';
import * as Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-batch';

@Component({
	selector: 'app-global-chat',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: './masterchat.html',
})
export class Masterchat implements AfterViewChecked {
	@ViewChild('chatWindow') chatWindowRef!: ElementRef;
	@ViewChild('messageInput') messageInputRef!: ElementRef;
	messages: { role: string; content: string }[] = [];
	query = '';
	private lastMessageCount = 0;
	loading = false;
	sessionId: string | null = null;
	docUrl = environment.docBase;
	apiUrl = environment.apiBase;
	matches: { doc_id: string; doc_name: string; text: string; link: string; score: number }[] = [];

	constructor(private http: HttpClient) { }

	ngAfterViewChecked(): void {
		if (this.lastMessageCount !== this.messages.length) {
			this.scrollToBottom();
			this.lastMessageCount = this.messages.length;
		}
	}
	private scrollToBottom(): void {
		try {
			this.chatWindowRef.nativeElement.scrollTop = this.chatWindowRef.nativeElement.scrollHeight;
			this.messageInputRef.nativeElement.focus();
		} catch { }
	}


	formatMessage(text: string): string {
		if (!text) return '';
		const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
		const codeBlocks: string[] = [];
		let i = 0;
		const temp = text.replace(codeBlockRegex, (_, lang, code) => {
			codeBlocks[i] = `<pre><code class=\"language-${lang || 'plaintext'}\">${code
				.replace(/&/g, '&amp;')
				.replace(/</g, '&lt;')
				.replace(/>/g, '&gt;')}</code></pre>`;
			return `%%CODEBLOCK_${i++}%%`;
		});
		const escaped = temp
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/\n/g, '<br>');
		return escaped.replace(/%%CODEBLOCK_(\d+)%%/g, (_, j) => codeBlocks[+j]);
	}

	async sendMessage(): Promise<void> {
		if (!this.query.trim()) return;
		this.matches = [];
		this.messages.push({ role: 'user', content: this.query });
		const text = this.query;
		this.query = '';
		this.loading = true;

		try {
			if (!this.sessionId) {
				const res = await firstValueFrom(
					this.http.post<{ session_id: string }>(
						`${environment.apiBase}/chat_session/start2`,
						{}
					)
				);
				this.sessionId = res.session_id;
				localStorage.setItem('globalSessionId', this.sessionId);
			}
			const res: any = await firstValueFrom(
				this.http.post<{ reply: string; matches: any[] }>(
					`${environment.apiBase}/chat_session/continue2`,
					{ session_id: this.sessionId, query: text }
				)
			);
			this.messages.push({ role: 'assistant', content: res.reply });
			if (Array.isArray(res.matched_docs)) {
				this.matches = res.matched_docs;
			}
			setTimeout(() => Prism.highlightAll(), 0);
		} catch (err) {
			console.error('Global chat session error', err);
		} finally {
			this.loading = false;
		}
	}


	async ngOnInit(): Promise<void> {
		// Restore session if present
		const savedSession = localStorage.getItem('globalSessionId');
		if (savedSession) {
			this.sessionId = savedSession;
			try {
				// Get the session history
				const res: any = await firstValueFrom(
					this.http.get(`${environment.apiBase}/chat_session/get_session/${this.sessionId}`)
				);
				if (Array.isArray(res.messages)) {
					// Replay the conversation to restore assistant replies and matches
					this.messages = [];
					this.matches = [];
					for (let i = 0; i < res.messages.length; i++) {
						const msg = res.messages[i];
						this.messages.push(msg);
						// If this is a user message and not the last message, replay to get assistant reply and matches
						if (msg.role === 'user') {
							// Only replay if next message is not assistant (i.e., incomplete history)
							if (!res.messages[i + 1] || res.messages[i + 1].role !== 'assistant') {
								// Call continue2 to get assistant reply and matches
								try {
									const replyRes: any = await firstValueFrom(
										this.http.post(`${environment.apiBase}/chat_session/continue2`, {
											session_id: this.sessionId,
											query: msg.content
										})
									);
									this.messages.push({ role: 'assistant', content: replyRes.reply });
									if (Array.isArray(replyRes.matched_docs)) {
										this.matches = replyRes.matched_docs;
									}
								} catch (err) {
									// Ignore errors for replay
								}
							}
						}
						// If this is the last assistant message, try to get matches for it
						if (i === res.messages.length - 1 && msg.role === 'assistant') {
							// Optionally, try to get matches for the last user query
							const lastUserMsg = res.messages.slice(0, i).reverse().find((m: { role: string; }) => m.role === 'user');
							if (lastUserMsg) {
								try {
									const replyRes: any = await firstValueFrom(
										this.http.post(`${environment.apiBase}/chat_session/continue2`, {
											session_id: this.sessionId,
											query: lastUserMsg.content
										})
									);
									if (Array.isArray(replyRes.matched_docs)) {
										this.matches = replyRes.matched_docs;
									}
								} catch (err) { }
							}
						}
					}
				}
			} catch (err) {
				// If session is invalid, clear it
				this.clearSession();
			}
		}
	}

	clearSession(): void {
		localStorage.removeItem('globalSessionId');
		this.sessionId = null;
		this.messages = [];
		this.matches = [];
	}
}