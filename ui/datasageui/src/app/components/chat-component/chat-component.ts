// src/app/components/chat-component/chat-component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Component({
	selector: 'app-chat-component',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: `chat-component.html`
})
export class ChatComponent implements OnInit {
	searchTerm: string = '';
	searchResults: any[] = [];
	pinnedDocs: any[] = [];
	selectedDocIds: string[] = [];
	messages: { role: string; content: string }[] = [];
	sessionId: string | null = null;
	loading: boolean = false;
	query: string = '';

	constructor(private http: HttpClient) { }

	ngOnInit(): void { }

	async searchDocuments() {
		const term = this.searchTerm.trim();
		if (!term) {
			this.searchResults = [];
			return;
		}
		const res = await this.http.get<any[]>(`${environment.apiBase}/document/search?query=${encodeURIComponent(term)}`).toPromise();
		this.searchResults = Array.isArray(res) ? res.filter(doc => doc.isIndexed) : [];
	}

	// get visibleDocs(): any[] {
	// 	const pinnedIds = new Set(this.pinnedDocs.map(d => d.id));
	// 	const unpinned = this.searchResults.filter(doc => !pinnedIds.has(doc.id));
	// 	return [...this.pinnedDocs, ...unpinned.slice(0, 5)];
	// }

	get visibleDocs(): any[] {
		const pinnedIds = new Set(this.pinnedDocs.map(d => d.id));
		const unpinned = this.searchResults.filter(doc => !pinnedIds.has(doc.id));
		return [...this.pinnedDocs, ...unpinned.slice(0, 5)];
	}


	togglePin(doc: any): void {
		const index = this.pinnedDocs.findIndex(d => d.id === doc.id);
		if (index > -1) {
			this.pinnedDocs.splice(index, 1);
		} else {
			this.pinnedDocs.push(doc);
		}
	}

	toggleSelect(docId: string, event: Event): void {
		const checked = (event.target as HTMLInputElement).checked;
		if (checked && !this.selectedDocIds.includes(docId)) {
			this.selectedDocIds.push(docId);
		} else if (!checked) {
			this.selectedDocIds = this.selectedDocIds.filter(id => id !== docId);
		}
	}

	getPinLabel(docId: string): string {
		return this.pinnedDocs.some(p => p.id === docId) ? 'Unpin' : 'Pin';
	}

	getMessageClass(role: string): string {
		return role === 'user' ? 'text-primary' : 'text-dark';
	}

	async sendMessage(): Promise<void> {
		const text = this.query.trim();
		if (!text || this.selectedDocIds.length === 0) {
			alert('Enter query and select documents');
			return;
		}

		if (!this.sessionId) {
			const res: any = await this.http.post(`${environment.apiBase}/chat_session/start`, {
				doc_ids: this.selectedDocIds
			}).toPromise();
			this.sessionId = res.session_id;
		}

		this.messages.push({ role: 'user', content: text });
		this.query = '';
		this.loading = true;

		const res: any = await this.http.post(`${environment.apiBase}/chat_session/continue`, {
			session_id: this.sessionId,
			query: text
		}).toPromise();

		//this.messages = res.messages;
		this.messages.push(...res.messages.filter((m: { role: string; }) => m.role !== 'user'));

		this.loading = false;
	}
}
