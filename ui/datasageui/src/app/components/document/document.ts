import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';

@Component({
	selector: 'app-document',
	standalone: true,
	imports: [
		CommonModule,
		ReactiveFormsModule
	],
	templateUrl: './document.html'
})
export class Document implements OnInit {
	private readonly http = inject(HttpClient);
	private readonly fb = inject(FormBuilder);
	private readonly API = environment.apiBase;

	uploadForm = this.fb.group({
		name: '',
		description: ''
	});

	selectedFile: File | null = null;
	uploading = signal(false);
	documents = signal<any[]>([]);
	search = signal('');
	searchResults: any;

	ngOnInit() {
		this.fetchDocs();
	}

	async indexDoc(docId: string): Promise<void> {
		try {
			const response: any = await firstValueFrom(
				this.http.post(`${this.API}/indexing/index`, {
					doc_ids: [docId]
				})
			);

			const result = response?.results?.find((r: any) => r.id === docId);

			if (result?.status === 'indexed') {
				await this.fetchDocs(); // reload all docs to reflect updated isIndexed
			} else {
				console.warn(`Indexing status for ${docId}:`, result?.status || 'unknown');
			}
		} catch (error) {
			console.error('Indexing failed:', error);
		}
	}



	onFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		this.selectedFile = input?.files?.[0] ?? null;
	}

	async uploadDoc() {
		if (!this.selectedFile || !this.uploadForm.valid) return;

		this.uploading.set(true);

		const formData = new FormData();
		formData.append('file', this.selectedFile);
		formData.append('name', this.uploadForm.value.name || '');
		formData.append('description', this.uploadForm.value.description || '');

		await this.http.post(`${this.API}/document/add`, formData).toPromise();

		this.uploadForm.reset();
		this.selectedFile = null;
		this.uploading.set(false);
		this.fetchDocs();
	}

	async fetchDocs() {
		const res = await this.http.get<any[]>(`${this.API}/document/list`).toPromise();
		this.documents.set(res ?? []);
	}

	delete(docId: string) {
		if (confirm('Delete this document?')) {
			this.http.post(`${this.API}/document/delete`, { doc_id: docId }).subscribe(() => {
				this.fetchDocs();
			});
		}
	}

	get filteredDocs() {
		const q = this.search().toLowerCase();
		return this.documents().filter(d => d.name.toLowerCase().includes(q));
	}

	formatDate(iso: string): string {
		return new Date(iso).toLocaleString();
	}
}
