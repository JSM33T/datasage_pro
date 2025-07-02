import { Component, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';
import { Collapse } from 'bootstrap';

@Component({
	selector: 'app-root',
	imports: [CommonModule, RouterModule, RouterLink],
	templateUrl: 'app.html',
	styleUrl: 'app.css',
})
export class App {
	token = signal(localStorage.getItem('token'));
	password = signal('');
	role = signal(localStorage.getItem('role') || 'user');
	loading = signal(false);
	error = signal('');

	constructor(private http: HttpClient, private router: Router) {
		if (!localStorage.getItem('token')) {
			this.router.navigateByUrl('/');
		}
	}

	ngAfterViewInit(): void {
		this.router.events.subscribe((event) => {
			if (event instanceof NavigationEnd) {
				const nav = document.getElementById('mainNav');
				if (nav && nav.classList.contains('show')) {
					new Collapse(nav).hide();
				}
			}
		});
	}

	confirmLogout() {
		localStorage.removeItem('token');
		this.router.navigateByUrl('/');
		location.reload();
	}
	login() {
		this.loading.set(true);
		this.error.set('');

		this.http
			.post<any>(`${environment.apiBase}/auth/login`, { password: this.password() })
			.subscribe({
				next: (response) => {
					localStorage.setItem('token', response.token);
					localStorage.setItem('role', response.role);

					this.token.set(response.token);
					this.role.set(response.role);

					this.loading.set(false);
					location.reload();
				},
				error: (err) => {
					this.loading.set(false);
					this.error.set(err.status === 401 ? 'Wrong password' : 'Login failed');
				},
			});
	}


	logout() {
		localStorage.removeItem('token');
		this.token.set(null);
		window.location.href = '/';
	}

}
