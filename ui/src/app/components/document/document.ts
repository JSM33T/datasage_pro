import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';
import * as bootstrap from 'bootstrap';

@Component({
	selector: 'app-document',
	standalone: true,
	imports: [
		CommonModule,
		ReactiveFormsModule
	],
	templateUrl: './document.html',
	styles: [`
		.description-text {
			display: -webkit-box;
			-webkit-line-clamp: 2;
			-webkit-box-orient: vertical;
			overflow: hidden;
			text-overflow: ellipsis;
			line-height: 1.4;
			max-height: 2.8em;
			font-size: 0.9rem;
		}
		
	   .card {
		   border: 1px solid #e9ecef;
		   border-radius: 4px;
	   }
	   
	   /* Remove card hover animation and extra shadow */
		
		.card-title {
			color: #495057;
			font-size: 1rem;
		}
		
		.btn-group-sm .btn {
			min-width: 40px;
			height: 32px;
			display: flex;
			align-items: center;
			justify-content: center;
			border-radius: 6px;
			transition: all 0.2s ease;
		}
		
		.btn-group-sm .btn:hover {
			transform: translateY(-1px);
		}
		
		.btn-group-sm .btn i {
			font-size: 14px;
		}
		
		.badge {
			font-size: 0.75rem;
			padding: 0.25em 0.5em;
		}
		
		.card-title i {
			opacity: 0.7;
		}
		
		@media (max-width: 768px) {
			.btn-group-sm {
				flex-direction: column !important;
				width: 100%;
				gap: 0.25rem !important;
			}
			
			.btn-group-sm .btn {
				width: 100%;
				min-width: auto;
			}
			
			.col-lg-4.col-md-5.col-12 {
				margin-top: 1rem;
			}
		}
		
		@media (min-width: 769px) {
			.btn-group-sm {
				flex-direction: row;
			}
		}
		
		.pagination .page-link {
			color: #6c757d;
			border: 1px solid #dee2e6;
			padding: 0.5rem 0.75rem;
		}
		
		.pagination .page-link:hover {
			background-color: #e9ecef;
			border-color: #dee2e6;
		}
		
		.pagination .page-link i {
			font-size: 0.9rem;
		}
		
		.pagination .page-item.active .page-link {
			background-color: #0d6efd;
			border-color: #0d6efd;
			color: white;
		}
		
		.pagination .page-item.disabled .page-link {
			color: #6c757d;
			background-color: #fff;
			border-color: #dee2e6;
		}
	`]
})
export class Document implements OnInit {
	private readonly http = inject(HttpClient);
	private readonly fb = inject(FormBuilder);
	private readonly API = environment.apiBase;

	// Signals and form setup
	currentPage = signal(1);
	pageSize = 10;
	totalDocs = signal(0);
	uploading = signal(false);
	indexingDocId = signal<string | null>(null);
	search = signal('');
	documents = signal<any[]>([]);
	role = localStorage.getItem('role') || 'user';

	uploadForm = this.fb.group({
		name: '',
		description: ''
	});

	selectedFile: File | null = null;

	ngOnInit() {
		this.fetchDocs();
	}
	get totalPages(): number {
		return Math.ceil(this.totalDocs() / this.pageSize);
	}

	// Helper method for pagination display
	getEndIndex(): number {
		return Math.min(this.currentPage() * this.pageSize, this.totalDocs());
	}

	// Fetch documents with pagination and search
	async fetchDocs() {
		const adminToken = localStorage.getItem('token') || '';
		const res: any = await this.http
			.get(`${this.API}/document/list`, {
				params: {
					page: this.currentPage().toString(),
					page_size: this.pageSize.toString(),
					search: this.search() || ''
				},
				headers: { Authorization: adminToken }
			})
			.toPromise();

		// Defensive assignment to ensure array
		this.documents.set(Array.isArray(res.items) ? res.items : []);
		this.totalDocs.set(res.total ?? 0);
	}


	// Pagination controls
	changePage(page: number) {
		if (page < 1) return;
		this.currentPage.set(page);
		this.fetchDocs();
	}

	// Search input change
	onSearchChange(event: Event) {
		const value = (event.target as HTMLInputElement).value;
		this.search.set(value);
		this.currentPage.set(1);
		this.fetchDocs();
	}

	// Index a document
	async indexDoc(docId: string): Promise<void> {
		this.indexingDocId.set(docId);
		const adminToken = localStorage.getItem('token') || '';
		try {
			const response: any = await firstValueFrom(
				this.http.post(
					`${this.API}/indexing/index`,
					{ doc_ids: [docId] },
					{ headers: { Authorization: adminToken } }
				)
			);

			const result = response?.results?.find((r: any) => r.id === docId);

			if (result?.status === 'indexed') {
				await this.fetchDocs();
			}
		} catch (error) {
			console.error('Indexing failed:', error);
		}
		this.indexingDocId.set(null);
	}

	// Reindex a document
	async reindexDoc(docId: string): Promise<void> {
		this.indexingDocId.set(docId);
		const adminToken = localStorage.getItem('token') || '';
		try {
			const response: any = await firstValueFrom(
				this.http.post(
					`${this.API}/document/reindex`,
					{ doc_ids: [docId] },
					{ headers: { Authorization: adminToken } }
				)
			);

			const result = response?.results?.find((r: any) => r.id === docId);

			if (result?.status === 'indexed') {
				await this.fetchDocs();
			}
		} catch (error) {
			console.error('Re-indexing failed:', error);
		}
		this.indexingDocId.set(null);
	}

	// Upload document
	async uploadDoc_OLD() {
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
	// Upload document
	async uploadDoc() {
		if (!this.selectedFile || !this.uploadForm.valid) return;

		this.uploading.set(true);

		const formData = new FormData();
		formData.append('file', this.selectedFile);
		formData.append('name', this.uploadForm.value.name || '');
		formData.append('description', this.uploadForm.value.description || '');

		const adminToken = localStorage.getItem('token') || '';

		try {
			// Get response with id
			const uploadResponse: any = await this.http.post(
				`${this.API}/document/add`,
				formData,
				{ headers: { Authorization: adminToken } }
			).toPromise();

			this.uploadForm.reset();
			this.selectedFile = null;

			// Clear file input element
			const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
			if (fileInput) fileInput.value = '';

			this.showToast('Document uploaded, indexing in progress.');

			// Refresh documents first
			this.fetchDocs();

			// Call indexing API for uploaded document in background
			if (uploadResponse?.id) {
				this.indexDoc(uploadResponse.id);
			}
		} catch (error: any) {
			console.error('Upload failed:', error);
			const errMsg = error?.error?.error || 'Upload failed.';
			this.showToast(errMsg, true);
		} finally {
			this.uploading.set(false);
		}
	}

	// Upload document
	async uploadDoc_OLD2() {
		if (!this.selectedFile || !this.uploadForm.valid) return;

		this.uploading.set(true);

		const formData = new FormData();
		formData.append('file', this.selectedFile);
		formData.append('name', this.uploadForm.value.name || '');
		formData.append('description', this.uploadForm.value.description || '');

		try {
			await this.http.post(`${this.API}/document/add`, formData).toPromise();
			this.uploadForm.reset();
			this.selectedFile = null;

			// Clear file input element
			const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
			if (fileInput) fileInput.value = '';

			this.showToast('Document uploaded successfully.');
			this.fetchDocs();
		} catch (error: any) {
			console.error('Upload failed:', error);
			const errMsg = error?.error?.error || 'Upload failed.';
			this.showToast(errMsg, true);
		} finally {
			this.uploading.set(false);
		}
	}



	// Delete document
	delete(docId: string) {
		if (confirm('Delete this document?')) {
			const adminToken = localStorage.getItem('token') || '';
			this.http.post(
				`${this.API}/document/delete`,
				{ doc_id: docId },
				{ headers: { Authorization: adminToken } }
			).subscribe(() => {
				this.fetchDocs();
			});
		}
	}

	// Download document
	download(doc: any) {
		// Use the new API endpoint for download
		const adminToken = localStorage.getItem('token') || '';
		const url = `${this.API}/document/download/${doc.id}/${encodeURIComponent(doc.filename)}`;
		// Create a hidden link to set the Authorization header
		fetch(url, {
			headers: { Authorization: adminToken }
		})
			.then(response => {
				if (!response.ok) throw new Error('Download failed');
				return response.blob();
			})
			.then(blob => {
				const link = document.createElement('a');
				link.href = window.URL.createObjectURL(blob);
				link.download = doc.filename;
				document.body.appendChild(link);
				link.click();
				document.body.removeChild(link);
			})
			.catch(() => {
				this.showToast('Download failed.', true);
			});
	}

	// File change event
	onFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		this.selectedFile = input?.files?.[0] ?? null;
	}

	// Format ISO date to readable string with consistent DD-MM-YYYY format
	formatDate(iso: string): string {
		if (!iso) return '';

		const date = new Date(iso);

		// Check if date is valid
		if (isNaN(date.getTime())) return '';

		const day = date.getDate().toString().padStart(2, '0');
		const month = (date.getMonth() + 1).toString().padStart(2, '0');
		const year = date.getFullYear();
		let hours = date.getHours();
		const minutes = date.getMinutes().toString().padStart(2, '0');
		const seconds = date.getSeconds().toString().padStart(2, '0');
		const ampm = hours >= 12 ? 'PM' : 'AM';
		hours = hours % 12;
		hours = hours ? hours : 12; // the hour '0' should be '12'
		const hoursStr = hours.toString().padStart(2, '0');

		return `${day}-${month}-${year} ${hoursStr}:${minutes}:${seconds} ${ampm}`;
	}

	get filteredDocs() {
		const docs = this.documents();
		if (!Array.isArray(docs)) return [];
		return docs;
	}

	// Show bootstrap toast
	showToast(message: string, isError: boolean = false) {
		const toastEl = document.getElementById('indexToast');
		if (toastEl) {
			toastEl.querySelector('.toast-body')!.textContent = message;
			toastEl.classList.remove('text-bg-success', 'text-bg-danger');
			toastEl.classList.add(isError ? 'text-bg-danger' : 'text-bg-success');
			const toast = new bootstrap.Toast(toastEl);
			toast.show();
		}
	}
}
