// src/app/document-management.component.ts
import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Component({
	selector: 'app-document-management',
	standalone: true,
	imports: [CommonModule, ReactiveFormsModule],
	templateUrl: './document-management-component.html'
})
export class DocumentManagementComponent implements OnInit {
	private readonly http = inject(HttpClient);
	private readonly fb = inject(FormBuilder);
	private readonly API_BASE = environment.apiBase;

	uploadForm = this.fb.group({
		name: '',
		description: ''
	});

	selectedFile: File | null = null;
	documents = signal<any[]>([]);
	selectedIds = new Set<string>();
	uploading = signal(false);
	indexingIds = new Set<string>();

	ngOnInit() {
		this.fetchDocs();
	}

	onFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files?.length) this.selectedFile = input.files[0];
	}

	async uploadDoc() {
		if (!this.selectedFile) return;
		this.uploading.set(true);

		const formData = new FormData();
		formData.append('file', this.selectedFile);
		formData.append('name', this.uploadForm.value.name || '');
		formData.append('description', this.uploadForm.value.description || '');

		await this.http.post(`${this.API_BASE}/document/add`, formData).toPromise();
		this.uploadForm.reset();
		this.selectedFile = null;
		this.uploading.set(false);
		this.fetchDocs();
	}

	async fetchDocs() {
		const docs = await this.http.get<any[]>(`${this.API_BASE}/document/list`).toPromise();
		this.documents.set(docs ?? []);
	}

	async indexDoc(docId: string) {
		this.indexingIds.add(docId);
		await this.http.post(`${this.API_BASE}/indexing/index`, { doc_ids: [docId] }).toPromise();
		this.indexingIds.delete(docId);
		this.fetchDocs();
	}

	toggleSelect(docId: string, event: Event) {
		const checked = (event.target as HTMLInputElement).checked;
		checked ? this.selectedIds.add(docId) : this.selectedIds.delete(docId);
	}

	getSelectedDocIds(): string[] {
		return Array.from(this.selectedIds);
	}
}
